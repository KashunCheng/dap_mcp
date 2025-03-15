import json
from dap_types import Variable, Scope
from pydantic import BaseModel
from typing import Optional, Tuple, Any, Callable


def is_plain_object(obj: Any) -> bool:
    if obj is None:
        return True
    if isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, list):
        return all(is_plain_object(item) for item in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and is_plain_object(v) for k, v in obj.items())
    return False


def try_dump_base_model(
    model: Any, fallback: Callable[[Any], str] = str
) -> str | list[str] | None:
    if model is None:
        return None
    if isinstance(model, BaseModel):
        return json.dumps(model.model_dump(exclude_none=True))
    if isinstance(model, list):
        if not is_plain_object(model):
            raise ValueError("Cannot dump a list of non-plain objects")
        return json.dumps(model)
    if isinstance(model, dict):
        if not is_plain_object(model):
            raise ValueError("Cannot dump a dict of non-plain objects")
        return json.dumps(
            {
                k: try_dump_base_model(v, fallback)
                for k, v in filter(lambda kv: kv[1] is not None, model.items())
            }
        )
    return fallback(model)


def render_xml(tag: str, content: str | list[str] | None, **attrs) -> str:
    filtered_attr_tuples = [(k, v) for k, v in attrs.items() if v is not None]
    attr_str = " ".join([f'{k}="{v}"' for k, v in filtered_attr_tuples])
    tag_with_attr_str = f"{tag} {attr_str}" if attr_str != "" else tag
    if content:
        if isinstance(content, list):
            content = "\n".join(content)
        return f"<{tag_with_attr_str}>{content}</{tag}>"
    return f"<{tag_with_attr_str}/>"


def render_variable(variable: Variable) -> str:
    return render_xml("variable", None, **variable.model_dump(exclude_none=True))


def render_scope(scope: Scope, variables: list[Variable]) -> str:
    return render_xml(
        "scope",
        [render_variable(variable) for variable in variables],
        name=scope.name,
        line=scope.line,
        column=scope.column,
    )


def render_table(active_id: Optional[int], lines: list[Tuple[int, str]]) -> str:
    # print lines with active line marked
    # active_id is the index of the active line in lines.
    # lines: [(line_number, line_content)].
    # If active_id equals to the index of the line in lines, that line is active.
    # It will print in the following format:
    # {line_number} -> {line_content}
    # {line_number}    {line_content} (otherwise)
    if len(lines) == 0:
        return ""
    max_line_number = max([line_number for line_number, _ in lines])
    max_line_number_length = len(str(max_line_number))
    formatted_lines = []
    for line_number, line_content in lines:
        if line_number == active_id:
            formatted_lines.append(
                f"{line_number:>{max_line_number_length}} -> {line_content}"
            )
        else:
            formatted_lines.append(
                f"{line_number:>{max_line_number_length}}    {line_content}"
            )
    return "\n" + "\n".join(formatted_lines) + "\n"
