# dap-mcp

**dap-mcp** is an implementation of the [Model Context Protocol (MCP)](https://example.com/mcp-spec) tailored for managing Debug Adapter Protocol (DAP) sessions. MCP provides a standardized framework to optimize and extend the context window of large language models, and in this project, it is used to enhance and streamline debugging workflows.

## Features

- **Debug Adapter Protocol Integration:** Interact with debuggers using a standardized protocol.
- **MCP Framework:** Leverage MCP to optimize context and enhance debugging workflows.
- **Rich Debugging Tools:** Set, list, and remove breakpoints; control execution (continue, step in/out/next); evaluate expressions; change stack frames; and view source code.
- **Flexible Configuration:** Customize debugger settings, source directories, and other parameters via a JSON configuration file.
## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (optional, for running the server)

### Installing and Running the Server

Install **dap-mcp** and its dependencies:

```bash
pip install dap-mcp
python -m dap_mcp --config config.json

# Or, if you have uv installed
uvx dap-mcp@latest --config config.json
```

## Configuration

The project uses a JSON configuration file (e.g., `.config.json`) to specify debugger settings and source directories. An example configuration:

```json5
{
  "type": "debugpy",
  "debuggerPath": "/path/to/python/with/debugpy",
  "debuggerArgs": [
    "-m",
    "debugpy.adapter"
  ],
  // source directories for resolving file paths
  // if you always use absolute paths, you can omit this
  "sourceDirs": [
    "/path/to/source/code"
  ],
  // debugger-specific settings start here
  // configurations for debugpy can be found at
  // https://github.com/microsoft/debugpy/wiki/Debug-configuration-settings
  
  // you can use "program" instead of "module" to specify the program to debug
  "module": "pytest",
  // the python executable to use to run the debuggee
  "python": ["/path/to/python"],
  "cwd": "/path/to/working/directory"
}
```

This configuration informs the debugger about:
- The path to the debugger executable and its arguments.
- The source directories for resolving file paths during breakpoint operations.
- Other settings (such as module, working directory, and interpreter path) necessary for launching the debuggee.

## Available Tools

The project exposes several tools that can be invoked via the MCP framework:

- **launch:** Launch the debuggee program.
- **set_breakpoint:** Set a breakpoint at a specified file and line (with an optional condition).
- **remove_breakpoint:** Remove a breakpoint from a specified file and line.
- **list_all_breakpoints:** List all breakpoints currently set in the debugger.
- **continue_execution:** Continue program execution after hitting a breakpoint.
- **step_in:** Step into a function call.
- **step_out:** Step out of the current function.
- **next:** Step over to the next line of code.
- **evaluate:** Evaluate an expression in the current debugging context.
- **change_frame:** Switch to a different stack frame.
- **view_file_at_line:** View source code around a specified line (using the last provided file if none is specified).
- **terminate:** Terminate the debugging session.

These tools provide XML-rendered output for integration with MCP clients.

## Extending with Other DAP Servers

To support additional DAP servers, you can simply add a new DAP-specific configuration class in the `dap_mcp/config.py` file. All DAP configurations extend from the base `DAPConfig` class. Each new subclass should:
  
- Define a unique `type` value (using a `Literal`) to act as a discriminator.
- Include any additional fields or settings specific to that debugger.

For example, to add support for a hypothetical DAP server called "mydap", you might add:

```python
class MyDAP(DAPConfig):
    type: Literal["mydap"]
    # Add any additional settings for MyDAP here
    customSetting: Optional[str] = Field(
        None, description="A custom setting for MyDAP."
    )
```

After creating your new configuration class, update the union type used for debugger-specific configurations by including your new class. For example:

```python
DebuggerSpecificConfig = Annotated[Union[DebugPy, MyDAP], Field(..., discriminator="type")]
```

Now, when you supply a configuration JSON with `"type": "mydap"`, it will be parsed and validated using your new `MyDAP` class, and your DAP server extension will be fully integrated.

### Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Write tests and ensure all checks pass.
4. Submit a pull request.

Please follow the coding guidelines and include appropriate tests with your changes.

## License

This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for details.