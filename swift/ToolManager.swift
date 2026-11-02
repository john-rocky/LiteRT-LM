// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

/// Manages the discovery, schema generation, and execution of tools.
public final class ToolManager: Sendable {

  /// A map of tool names to the tools.
  private let toolRegistry: [String: ToolProtocol]

  /// The JSON schema string representing all registered tools.
  /// This is computed once during initialization.
  public let toolsJsonDescription: String

  /// Initializes the ToolManager with a list of tool instances.
  ///
  /// - Parameter tools: A list of instantiated tools (e.g. `[GetWeatherTool()]`).
  ///   These instances are used *only* for schema generation. Execution uses new instances.
  public convenience init(tools: [Tool]) {
    // A `Tool`'s own description, built from its type, always has a `name` string, so
    // `entry(for:)` does not throw for it. A `Tool` that overrides `getToolDescriptionJsonString()`
    // with a description `entry(for:)` rejects is left out, where `init(tools: [ToolProtocol])`
    // throws.
    self.init(entries: tools.compactMap { try? Self.entry(for: $0) })
  }

  /// Initializes the ToolManager with tools of any kind: `Tool` instances, `OpenApiTool`s, and any
  /// other `ToolProtocol`.
  ///
  /// - Parameter tools: The tools, in the order the model reads them. Each tool's description is
  ///   used as written, inside the `{"type": "function", "function": ...}` object the native
  ///   library expects.
  /// - Throws: `LiteRTLMError.tool(.invalidDescription)` if a tool's description is not a JSON
  ///   object with a `name` string.
  public convenience init(tools: [ToolProtocol]) throws {
    self.init(entries: try tools.map { try Self.entry(for: $0) })
  }

  private init(entries: [Entry]) {
    var toolRegistry: [String: ToolProtocol] = [:]
    for entry in entries {
      toolRegistry[entry.name] = entry.tool
    }
    self.toolRegistry = toolRegistry
    // Concatenated, not re-serialized: JSONSerialization does not keep the key order.
    let wrapped = entries.map { #"{"type":"function","function":"# + $0.description + "}" }
    self.toolsJsonDescription = "[" + wrapped.joined(separator: ",") + "]"
  }

  /// A tool with its description and the `name` in it.
  private struct Entry {
    let name: String
    let description: String
    let tool: ToolProtocol
  }

  /// The entry of a tool whose description is a JSON object with a `name` string.
  private static func entry(for tool: ToolProtocol) throws -> Entry {
    let description = tool.getToolDescriptionJsonString()
    guard
      let object = try? JSONSerialization.jsonObject(with: Data(description.utf8))
        as? [String: Any]
    else {
      throw LiteRTLMError.tool(.invalidDescription("Not a JSON object: \(description)"))
    }
    guard let name = object["name"] as? String else {
      throw LiteRTLMError.tool(.invalidDescription("No \"name\" string: \(description)"))
    }
    return Entry(name: name, description: description, tool: tool)
  }

  /// Executes a tool based on the name and JSON arguments provided by the model.
  ///
  /// - Parameters:
  ///   - name: The name of the tool to execute.
  ///   - arguments: The arguments as a dictionary.
  /// - Returns: The result of the tool execution. This will be a JSON-compatible type (String, Number, Bool, Array, Dictionary).
  ///   For an `OpenApiTool`, it is the JSON string its `execute(paramsJsonString:)` returned.
  public func execute(name: String, arguments: [String: Any]) async throws -> Any {
    guard let tool = toolRegistry[name] else {
      throw LiteRTLMError.tool(.notFound(name: name))
    }

    // Normalize the result to ensure it is strictly JSON-compatible (mimicking Android logic)
    return normalizeResult(try await tool.execute(arguments: arguments))
  }

  /// Helper to recursively convert any result into a JSON-Serialization friendly format.
  /// - Collections: Recursively normalized.
  /// - Primitives: Passed through.
  /// - Custom Objects: Converted to String.
  private func normalizeResult(_ value: Any) -> Any {
    // Handle Dictionaries
    if let dict = value as? [String: Any] {
      var normalizedDict: [String: Any] = [:]
      for (key, val) in dict {
        normalizedDict[key] = normalizeResult(val)
      }
      return normalizedDict
    }

    // Handle Arrays
    if let array = value as? [Any] {
      return array.map { normalizeResult($0) }
    }

    // Handle Primitives (String, Number, Bool)
    if value is String || value is Int || value is Double || value is Float || value is Bool {
      return value
    }

    // Handle Void
    if value is Void {
      return ""
    }

    // Fallback for Custom Objects
    return String(describing: value)
  }
}
