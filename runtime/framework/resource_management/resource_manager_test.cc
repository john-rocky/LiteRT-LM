// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/resource_management/resource_manager.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

// Returns the inputs of a prefill of `token_ids`.
ExecutorInputs TextInputs(absl::Span<const int> token_ids) {
  auto token_ids_buffer = CopyToTensorBuffer<int>(
      token_ids, {1, static_cast<int>(token_ids.size())});
  ExecutorInputs inputs;
  inputs.SetTextData(ExecutorTextData(std::move(*token_ids_buffer)));
  return inputs;
}

// A fake executor that reports executor stats and counts stat resets.
class StatsFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit StatsFakeLlmExecutor(int* num_resets)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        num_resets_(num_resets) {
    stats_.AccumulatePlainStep(absl::Milliseconds(10));
  }

  std::optional<ExecutorStats> GetExecutorStats() const override {
    return stats_;
  }
  void ResetExecutorStats() override { ++*num_resets_; }

 private:
  ExecutorStats stats_{.module_name = std::string(kLlmModuleName)};
  int* num_resets_;
};

// Tasks::Decode receives the `LockedLlmExecutor` wrapper rather than the
// underlying executor, so the stats APIs must be forwarded through it.
TEST(ResourceManagerTest, AcquiredExecutorForwardsExecutorStats) {
  int num_resets = 0;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<StatsFakeLlmExecutor>(&num_resets),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());

  std::optional<ExecutorStats> stats = executor->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->module_name, kLlmModuleName);
  EXPECT_EQ(stats->plain_steps(), 1);

  executor->ResetExecutorStats();
  EXPECT_EQ(num_resets, 1);
}

// A fake executor that reports a given draft verification support and accepts
// every draft that it verifies.
class DraftVerificationFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit DraftVerificationFakeLlmExecutor(absl::Status support)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        support_(std::move(support)) {}

  absl::Status CheckDraftVerificationSupport() const override {
    return support_;
  }

  absl::StatusOr<DraftVerificationResult> VerifyDraft(
      absl::Span<const int> draft_token_ids,
      const ExecutorDecodeParams& decode_params) override {
    return DraftVerificationResult{
        .tokens =
            std::vector<int>(draft_token_ids.begin(), draft_token_ids.end()),
        .num_accepted = static_cast<int>(draft_token_ids.size())};
  }

 private:
  absl::Status support_;
};

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerificationSupport) {
  for (const absl::Status& support :
       {absl::OkStatus(), absl::FailedPreconditionError("No verify.")}) {
    SCOPED_TRACE(support.ToString());
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr,
            std::make_unique<DraftVerificationFakeLlmExecutor>(support),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                         resource_manager->AcquireExecutor());
    EXPECT_EQ(executor->CheckDraftVerificationSupport(), support);
  }
}

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerification) {
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<DraftVerificationFakeLlmExecutor>(absl::OkStatus()),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());
  EXPECT_THAT(executor->VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
              IsOkAndHolds(FieldsAre(ElementsAre(3, 1, 4), 3)));
}

TEST(ResourceManagerTest, ExecutorsDontSupportDraftVerificationByDefault) {
  FakeLlmExecutor executor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                           /*decode_tokens_set=*/{});
  // The defaults of `LlmExecutor`.
  EXPECT_THAT(executor.LlmExecutor::CheckDraftVerificationSupport(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      executor.LlmExecutor::VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ResourceManagerTest, ExecutorsDontVerifyDraftsByDefault) {
  FakeLlmExecutor executor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                           /*decode_tokens_set=*/{});
  executor.SetDraftVerificationSupport(absl::OkStatus());
  // The default of `LlmExecutor` for an executor that claims support.
  EXPECT_THAT(
      executor.LlmExecutor::VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kUnimplemented));
}

// How the resource manager has installed contexts into an executor.
struct ContextSwitchCounts {
  int cloned_contexts = 0;
  int released_contexts = 0;
  int released_with_save_state = 0;
  int released_without_save_state = 0;
  int restored_contexts = 0;
  int updated_runtime_states = 0;
  std::vector<int> prefill_start_steps;
};

// A fake executor that counts context installs and releases.
class ContextSwitchFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit ContextSwitchFakeLlmExecutor(
      ContextSwitchCounts* counts,
      std::vector<std::vector<int>> prefill_tokens_set = {{1, 2, 3}})
      : FakeLlmExecutor(/*vocab_size=*/16, prefill_tokens_set,
                        /*decode_tokens_set=*/{{4}}),
        counts_(counts) {}

  using FakeLlmExecutor::Prefill;
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    counts_->prefill_start_steps.push_back(GetCurrentStep().value_or(-1));
    return FakeLlmExecutor::Prefill(inputs, prefill_params);
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    ++counts_->cloned_contexts;
    return MakeContext();
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> ReleaseContext(
      bool save_state) override {
    ++counts_->released_contexts;
    if (save_state) {
      ++counts_->released_with_save_state;
    } else {
      ++counts_->released_without_save_state;
    }
    return MakeContext();
  }

  using FakeLlmExecutor::RestoreContext;
  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> llm_context) override {
    ++counts_->restored_contexts;
    return SetCurrentStep(llm_context->runtime_state().current_step);
  }
  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return RuntimeConfig();
  }
  absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) override {
    return absl::OkStatus();
  }
  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return RuntimeState{.current_step = GetCurrentStep().value_or(0)};
  }
  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    ++counts_->updated_runtime_states;
    return SetCurrentStep(runtime_state.current_step);
  }

 private:
  std::unique_ptr<LlmContext> MakeContext() const {
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(),
        std::make_unique<RuntimeState>(
            RuntimeState{.current_step = GetCurrentStep().value_or(0)}));
  }

  ContextSwitchCounts* counts_;
};

// Clones of a session past step 0 share its processed context, so the resource
// manager switches between them with `UpdateRuntimeState()` rather than
// `RestoreContext()`. Executors that keep per-context state must treat both as
// a context switch (b/578989824).
TEST(ResourceManagerTest, SwitchesToClonesByUpdatingRuntimeState) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(&counts),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> root,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(root));
    EXPECT_EQ(counts.restored_contexts, 1);
    // Moves the root past step 0, since a clone of a session at step 0 gets its
    // own processed context.
    ASSERT_OK(executor->Prefill(TextInputs({1, 2, 3})));
  }

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone_1,
                       resource_manager->CloneContextHandler(root));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone_2,
                       resource_manager->CloneContextHandler(root));
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(clone_1));
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(clone_2));
  EXPECT_EQ(counts.restored_contexts, 1);
  EXPECT_EQ(counts.updated_runtime_states, 2);
}

TEST(ResourceManagerTest, SkipsContextSaveWhenSwitchingFromReleasedHandler) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(&counts),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> session_1,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(session_1));
    ASSERT_OK(executor->SetCurrentStep(3));
  }
  EXPECT_EQ(counts.restored_contexts, 1);
  EXPECT_EQ(counts.released_contexts, 0);
  EXPECT_EQ(counts.cloned_contexts, 0);

  // Destroy session_1 before switching to session_2: the outgoing context has
  // no live owner or clone, so ResourceManager must release it without saving
  // its state and must not clone it.
  session_1.reset();

  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> session_2,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(session_2));
  EXPECT_EQ(counts.restored_contexts, 2);
  EXPECT_EQ(counts.released_contexts, 1);
  EXPECT_EQ(counts.released_without_save_state, 1);
  EXPECT_EQ(counts.released_with_save_state, 0);
  EXPECT_EQ(counts.cloned_contexts, 0);
}

TEST(ResourceManagerTest,
     PreservesContextWhenSwitchingFromReleasedHandlerWithLiveClone) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(&counts),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> session_1,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(session_1));
    ASSERT_OK(executor->SetCurrentStep(3));
  }
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone_1,
                       resource_manager->CloneContextHandler(session_1));

  // Destroy session_1 while clone_1 still shares its SharedProcessedContext.
  session_1.reset();

  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> session_2,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(session_2));
  EXPECT_EQ(counts.released_contexts, 1);
  EXPECT_EQ(counts.released_with_save_state, 1);
  EXPECT_TRUE(clone_1->shared_processed_context()->HasProcessedContext());
}

TEST(ResourceManagerTest,
     PassesSaveStateFalseAtStepZeroAndTrueAfterStepAdvances) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(&counts),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> session_1,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> session_2,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));

  // Acquire session_1 at step 0, then switch to session_2: save_state must be
  // false because session_1 is still at step 0.
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(session_1));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(session_2));
    EXPECT_EQ(counts.released_contexts, 1);
    EXPECT_EQ(counts.released_without_save_state, 1);
    EXPECT_EQ(counts.released_with_save_state, 0);
    ASSERT_OK(executor->SetCurrentStep(3));
  }

  // Switch back to session_1 while session_2 is at step 3: save_state must be
  // true.
  ASSERT_OK(resource_manager->AcquireExecutorWithContextHandler(session_1));
  EXPECT_EQ(counts.released_contexts, 2);
  EXPECT_EQ(counts.released_without_save_state, 1);
  EXPECT_EQ(counts.released_with_save_state, 1);
}

TEST(ResourceManagerTest, RotatesThreeLiveSessionsWithoutCloning) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(&counts),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  std::vector<std::shared_ptr<ContextHandler>> sessions;
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(
        std::shared_ptr<ContextHandler> session,
        resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
    sessions.push_back(std::move(session));
  }

  // Advances every session past step 0, then rotates through them twice.
  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < sessions.size(); ++i) {
      SCOPED_TRACE(absl::StrCat("round=", round, " session=", i));
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<LlmExecutor> executor,
          resource_manager->AcquireExecutorWithContextHandler(sessions[i]));
      if (round == 0) {
        ASSERT_OK(executor->SetCurrentStep(i + 1));
      } else {
        // Each session gets back the step it left at.
        EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(i + 1));
      }
    }
  }

  // 9 acquisitions: the first loads into an empty executor and the other 8
  // each release the outgoing session, which is always past step 0, so every
  // release saves (moves) its state and none of them clones.
  EXPECT_EQ(counts.restored_contexts, 9);
  EXPECT_EQ(counts.released_contexts, 8);
  EXPECT_EQ(counts.released_without_save_state, 0);
  EXPECT_EQ(counts.released_with_save_state, 8);
  EXPECT_EQ(counts.cloned_contexts, 0);
}

absl::StatusOr<ExecutorInputs> MakeTextInputs(absl::Span<const int> tokens) {
  auto buffer = CopyToTensorBuffer<int>(
      tokens, {1, static_cast<int>(tokens.size())});
  if (!buffer.HasValue()) {
    return absl::InternalError(buffer.Error().Message());
  }
  return ExecutorInputs(ExecutorTextData(std::move(*buffer)), std::nullopt,
                        std::nullopt);
}

TEST(ResourceManagerTest,
     SyncsExecutorStepBeforePrefillingSuffixAfterFullPrefixMatch) {
  ContextSwitchCounts counts;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextSwitchFakeLlmExecutor>(
              &counts, std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5, 6}}),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> root,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(root));
    ASSERT_OK_AND_ASSIGN(ExecutorInputs inputs, MakeTextInputs({1, 2}));
    ASSERT_OK(executor->Prefill(inputs, ExecutorPrefillParams()));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(2));
  }

  // Clone `root` at step 2, then advance `root` to step 4 so the shared
  // `ProcessedTokens` holds {1, 2, 3, 4} while `clone` remains at step 2.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone,
                       resource_manager->CloneContextHandler(root));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(root));
    ASSERT_OK_AND_ASSIGN(ExecutorInputs inputs, MakeTextInputs({3, 4}));
    ASSERT_OK(executor->Prefill(inputs, ExecutorPrefillParams()));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(4));
  }

  // Acquire `clone` (restoring runtime current_step to 2) and prefill
  // {3, 4, 5, 6}: `RemoveMatchingTokens` strips {3, 4} up to step 4 ==
  // `processed_tokens->TokenCount()`, and the executor must be synced to step 4
  // before prefilling the remaining suffix {5, 6}.
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(clone));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(2));
    ASSERT_OK_AND_ASSIGN(ExecutorInputs inputs, MakeTextInputs({3, 4, 5, 6}));
    ASSERT_OK(executor->Prefill(inputs, ExecutorPrefillParams()));
    EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(6));
    ASSERT_OK_AND_ASSIGN(const ProcessedTokens* processed_tokens,
                         executor->GetProcessedTokens());
    EXPECT_THAT(processed_tokens->GetCopyOfTokens(),
                ElementsAre(ElementsAre(1, 2, 3, 4, 5, 6)));
  }

  EXPECT_THAT(counts.prefill_start_steps, ElementsAre(0, 2, 4));
  EXPECT_EQ(counts.cloned_contexts, 0);
}

// A fake executor that records the tokens of each prefill that reaches it.
class PrefillRecordingFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit PrefillRecordingFakeLlmExecutor(
      std::vector<std::vector<int>>* prefills)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{{1, 2, 3}},
                        /*decode_tokens_set=*/{{4}}),
        prefills_(prefills) {}

  using FakeLlmExecutor::Prefill;
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    ABSL_ASSIGN_OR_RETURN(const TensorBuffer* token_ids,
                          inputs.GetTextTokenIdsPtr());
    LITERT_ASSIGN_OR_RETURN(std::vector<int> ids,
                            CopyFromTensorBuffer<int>(*token_ids));
    prefills_->push_back(std::move(ids));
    return FakeLlmExecutor::Prefill(inputs, prefill_params);
  }

 private:
  std::vector<std::vector<int>>* prefills_;
};

// After a rewind to step 0, a prefill of the same tokens runs on all of them:
// the executor removed its processed tokens, so none are left to match.
TEST(ResourceManagerTest, PrefillAfterRewindToStepZeroRunsAllTokens) {
  std::vector<std::vector<int>> prefills;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<PrefillRecordingFakeLlmExecutor>(&prefills),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> handler,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LlmExecutor> executor,
      resource_manager->AcquireExecutorWithContextHandler(handler));
  ASSERT_OK(executor->Prefill(TextInputs({1, 2, 3})));
  ASSERT_OK(executor->Decode());

  ASSERT_OK(executor->SetCurrentStep(0));
  ASSERT_OK(executor->Prefill(TextInputs({1, 2, 3})));

  EXPECT_THAT(prefills,
              ElementsAre(ElementsAre(1, 2, 3), ElementsAre(1, 2, 3)));
  EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(3));
}

// A fake executor that keeps the runtime state and the processed tokens in the
// loaded context, as the LiteRT compiled model executors do, so that the
// resource manager can switch between contexts. As in those executors, a new
// sequence (`Reset()`, a rewind to step 0, or a context restored at step 0) has
// no processed tokens. A prefill appends its tokens, and a decode appends
// `kDecodedToken`.
class ContextFakeLlmExecutor : public FakeLlmExecutor {
 public:
  static constexpr int kDecodedToken = 9;

  ContextFakeLlmExecutor()
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        llm_context_(NewContext(RuntimeConfig())) {}

  using FakeLlmExecutor::Decode;
  using FakeLlmExecutor::Prefill;
  using FakeLlmExecutor::RestoreContext;

  absl::Status Prefill(const ExecutorInputs& inputs) override {
    return Prefill(inputs, ExecutorPrefillParams());
  }
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& prefill_params) override {
    ABSL_ASSIGN_OR_RETURN(const TensorBuffer* token_ids,
                          inputs.GetTextTokenIdsPtr());
    LITERT_ASSIGN_OR_RETURN(std::vector<int> ids,
                            CopyFromTensorBuffer<int>(*token_ids));
    ABSL_RETURN_IF_ERROR(processed_tokens().RollBackToStep(current_step()));
    processed_tokens().AddProcessedTokens(ids);
    current_step() += static_cast<int>(ids.size());
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params) override {
    ABSL_RETURN_IF_ERROR(processed_tokens().RollBackToStep(current_step()));
    processed_tokens().AddProcessedTokens({kDecodedToken});
    ++current_step();
    return std::vector<std::vector<int>>{{kDecodedToken}};
  }

  absl::StatusOr<int> GetCurrentStep() const override {
    return llm_context_->runtime_state().current_step;
  }

  absl::Status SetCurrentStep(int new_step) override {
    if (new_step < 0 || new_step > processed_tokens().TokenCount()) {
      return absl::InvalidArgumentError("The step is out of range.");
    }
    if (new_step == 0) {
      return Reset();
    }
    current_step() = new_step;
    return absl::OkStatus();
  }

  absl::Status Reset() override {
    current_step() = 0;
    return processed_tokens().RollBackToStep(0);
  }

  absl::StatusOr<const ProcessedTokens*> GetProcessedTokens() const override {
    return &processed_tokens();
  }

  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return llm_context_->runtime_config();
  }
  absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) override {
    llm_context_->runtime_config() = runtime_config;
    return absl::OkStatus();
  }
  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return llm_context_->runtime_state();
  }
  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    llm_context_->runtime_state() = runtime_state;
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      RuntimeConfig runtime_config) const override {
    return NewContext(std::move(runtime_config));
  }
  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    auto processed_context = std::make_unique<FakeProcessedContext>();
    processed_context->processed_tokens() = processed_tokens();
    return std::make_unique<LlmContext>(
        std::move(processed_context),
        std::make_unique<RuntimeConfig>(llm_context_->runtime_config()),
        std::make_unique<RuntimeState>(llm_context_->runtime_state()));
  }
  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> llm_context) override {
    llm_context_ = std::move(llm_context);
    if (llm_context_->runtime_state().current_step == 0) {
      return Reset();
    }
    return absl::OkStatus();
  }

 private:
  static std::unique_ptr<LlmContext> NewContext(RuntimeConfig runtime_config) {
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(std::move(runtime_config)),
        std::make_unique<RuntimeState>());
  }

  int& current_step() { return llm_context_->runtime_state().current_step; }
  ProcessedTokens& processed_tokens() const {
    return llm_context_->processed_context().processed_tokens();
  }

  // The loaded context.
  std::unique_ptr<LlmContext> llm_context_;
};

// A rewind to step 0 starts a new sequence in the processed context loaded in
// the executor. A clone that shares that processed context keeps its own copy.
TEST(ResourceManagerTest, RewindToStepZeroKeepsTheClonesContext) {
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextFakeLlmExecutor>(),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> original,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(original));
    ASSERT_OK(executor->Prefill(TextInputs({1, 2, 3})));
    ASSERT_OK(executor->Decode());
  }
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone,
                       resource_manager->CloneContextHandler(original));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(original));
    ASSERT_OK(executor->SetCurrentStep(0));
  }

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LlmExecutor> executor,
      resource_manager->AcquireExecutorWithContextHandler(clone));
  EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(4));
  ASSERT_OK_AND_ASSIGN(const ProcessedTokens* processed_tokens,
                       executor->GetProcessedTokens());
  EXPECT_THAT(
      processed_tokens->GetCopyOfTokens(),
      ElementsAre(ElementsAre(1, 2, 3, ContextFakeLlmExecutor::kDecodedToken)));
  EXPECT_OK(executor->Decode());
}

// A clone of a session at step 0 gets its own processed context, so that its
// prefill does not leave processed tokens in the original's.
TEST(ResourceManagerTest, CloneAtStepZeroGetsItsOwnContext) {
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<ContextFakeLlmExecutor>(),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ContextHandler> original,
      resource_manager->CreateContextHandler(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone,
                       resource_manager->CloneContextHandler(original));
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LlmExecutor> executor,
        resource_manager->AcquireExecutorWithContextHandler(clone));
    ASSERT_OK(executor->Prefill(TextInputs({1, 2, 3})));
  }

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LlmExecutor> executor,
      resource_manager->AcquireExecutorWithContextHandler(original));
  EXPECT_THAT(executor->GetCurrentStep(), IsOkAndHolds(0));
  ASSERT_OK_AND_ASSIGN(const ProcessedTokens* processed_tokens,
                       executor->GetProcessedTokens());
  EXPECT_EQ(processed_tokens->TokenCount(), 0);
}

}  // namespace
}  // namespace litert::lm
