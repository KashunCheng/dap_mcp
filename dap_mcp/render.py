def render_xml(tag: str, content: str | None, **attrs) -> str:
    filtered_attr_tuples = [(k, v) for k, v in attrs.items() if v is not None]
    attr_str = {" ".join([f'{k}="{v}"' for k, v in filtered_attr_tuples])}
    if content:
        return f"<{tag} {attr_str}>{content}</{tag}>"
    return f"<{tag} {attr_str}/>"
