from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Union, Annotated


class DebuggerViewConfig(BaseModel):
    showVariables: bool = Field(
        default=True, description="Show variables in the debugger view."
    )
    maxVariableExprLength: Optional[int] = Field(
        default=None, description="Maximum length of variable expressions."
    )


class BaseToolConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable or disable the tool.")


class GetLaunchConfigConfig(BaseToolConfig):
    pass


class LaunchConfig(BaseToolConfig):
    pass


class SetBreakpointConfig(BaseToolConfig):
    pass


class RemoveBreakpointConfig(BaseToolConfig):
    pass


class ViewFileAroundLineConfig(BaseToolConfig):
    pass


class ListAllBreakpointsConfig(BaseToolConfig):
    pass


class RemoveAllBreakpointsConfig(BaseToolConfig):
    pass


class ContinueExecutionConfig(BaseToolConfig):
    pass


class StepInConfig(BaseToolConfig):
    pass


class StepOutConfig(BaseToolConfig):
    pass


class NextConfig(BaseToolConfig):
    pass


class EvaluateConfig(BaseToolConfig):
    pass


class ChangeFrameConfig(BaseToolConfig):
    pass


class TerminateConfig(BaseToolConfig):
    pass


class DAPToolsConfig(BaseModel):
    debuggerView: DebuggerViewConfig = Field(
        default_factory=DebuggerViewConfig, description="Debugger view configuration."
    )
    getLaunchConfig: GetLaunchConfigConfig = Field(
        default_factory=GetLaunchConfigConfig
    )
    launch: LaunchConfig = Field(default_factory=LaunchConfig)
    setBreakpoint: SetBreakpointConfig = Field(default_factory=SetBreakpointConfig)
    removeBreakpoint: RemoveBreakpointConfig = Field(
        default_factory=RemoveBreakpointConfig
    )
    viewFileAroundLine: ViewFileAroundLineConfig = Field(
        default_factory=ViewFileAroundLineConfig
    )
    listAllBreakpoints: ListAllBreakpointsConfig = Field(
        default_factory=ListAllBreakpointsConfig
    )
    removeAllBreakpoints: RemoveAllBreakpointsConfig = Field(
        default_factory=RemoveAllBreakpointsConfig
    )
    continueExecution: ContinueExecutionConfig = Field(
        default_factory=ContinueExecutionConfig
    )
    stepIn: StepInConfig = Field(default_factory=StepInConfig)
    stepOut: StepOutConfig = Field(default_factory=StepOutConfig)
    next: NextConfig = Field(default_factory=NextConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    changeFrame: ChangeFrameConfig = Field(default_factory=ChangeFrameConfig)
    terminate: TerminateConfig = Field(default_factory=TerminateConfig)


class DAPConfig(BaseModel):
    type: str
    debuggerPath: str = Field(..., description="Path to the debugger executable.")
    debuggerArgs: List[str] = Field(
        [], description="List of arguments to pass to the debugger executable."
    )
    sourceDirs: List[str] = Field(
        default=[],
        description="List of source directories. Will be used to resolve source paths for set_breakpoint, remove_breakpoint, and view_file_around_line when given paths are relative.",
    )
    tools: DAPToolsConfig = Field(
        default_factory=DAPToolsConfig, description="Tool specific configuration."
    )


class DebugPy(DAPConfig):
    type: Literal["debugpy"]

    # Code Execution Settings
    module: Optional[str] = Field(
        None, description="Name of the module to be debugged."
    )
    program: Optional[str] = Field(None, description="Absolute path to the program.")
    code: Optional[str] = Field(
        None,
        description='Code to execute in string form. Example: "import debugpy;print(debugpy.__version__)"',
    )
    python: List[str] = Field(
        ...,
        description='Path python executable and interpreter arguments. Example: ["/usr/bin/python", "-E"].',
    )
    args: Optional[List[str]] = Field(
        None,
        description='Command line arguments passed to the program. Example: ["--arg1", "-arg2", "val", ...].',
    )
    console: Optional[
        Literal["internalConsole", "integratedTerminal", "externalTerminal"]
    ] = Field(
        "internalConsole",
        description='Sets where to launch the debug target. Supported values: ["internalConsole", "integratedTerminal", "externalTerminal"]. Default is "internalConsole".',
    )
    cwd: Optional[str] = Field(
        None,
        description="Absolute path to the working directory of the program being debugged.",
    )
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables defined as a key value pair."
    )

    # Debugger Settings
    django: bool = Field(
        False, description="When true enables Django templates. Default is false."
    )
    gevent: bool = Field(
        False,
        description="When true enables debugging of gevent monkey-patched code. Default is false.",
    )
    jinja: bool = Field(
        False,
        description="When true enables Jinja2 template debugging (e.g. Flask). Default is false.",
    )
    justMyCode: bool = Field(
        True,
        description="When true debug only user-written code. To debug standard library or anything outside of 'cwd' use false. Default is true.",
    )
    logToFile: bool = Field(
        False,
        description="When true enables logging of debugger events to a log file(s). Default is false.",
    )
    pathMappings: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Map of local and remote paths. Example: [{'localRoot': 'local path', 'remoteRoot': 'remote path'}].",
    )
    pyramid: bool = Field(
        False,
        description="When true enables debugging Pyramid applications. Default is false.",
    )
    redirectOutput: bool = Field(
        False,
        description="When true redirects output to debug console. Default is false.",
    )
    showReturnValue: bool = Field(
        False,
        description="Shows return value of functions when stepping. The return value is added to the response to Variables Request.",
    )
    stopOnEntry: bool = Field(
        False,
        description="When true debugger stops at first line of user code. When false debugger does not stop until breakpoint, exception or pause.",
    )
    subProcess: bool = Field(
        True,
        description="When true enables debugging multiprocess applications. Default is true.",
    )
    sudo: bool = Field(
        False,
        description="When true runs program under elevated permissions (on Unix). Default is false.",
    )


DebuggerSpecificConfig = Annotated[Union[DebugPy], Field(..., discriminator="type")]
