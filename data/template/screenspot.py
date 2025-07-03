_SCREENSPOT_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location."
# _SYSTEM_point = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
_SYSTEM_point = "The coordinate represents a clickable location [x, y] for an element."
_SYSTEM_bbox = "The coordinates represent a bounding box [x1, y1, x2, y2] for an element."
_SYSTEM_point_int = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 1 to 1000."
_SYSTEM_thing_grounding = """Given the screenshot and the instruction:
"{INSTRUCTION}"
Analyze carefully and identify the best location (point) that corresponds to the instruction.
First, output the selected point as a JSON array in the format:
```json
[x,y]
```
Second, explain your reasoning using a numbered list:
1. Visual appearance (e.g., icon, button color or shape)
2. Text content (e.g., label text)
3. Position (e.g., relative location in the interface)
4. Context (e.g., nearby elements or semantic grouping)

Only output the coordinates in a separate JSON block at the start."""

_SCREENSPOT_USER = '<|image_1|>{system}{element}'

def screenspot_to_qwen(element_name, image, xy_int=False, isbbox=False, think_grounding=False):
    transformed_data = []
    user_content = []

    if xy_int:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point_int
    else:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point
    if isbbox:
        system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_bbox

    '{system}<|image_1|>{element}'
    user_content.append({"type": "text", "text": system_prompt})
    user_content.append(image)
    user_content.append({"type": "text",  "text": element_name})
    
    if think_grounding:
        system_prompt = _SYSTEM_thing_grounding.format(INSTRUCTION=element_name)
        user_content = []
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append(image)    

    # question = _SCREENSPOT_USER.format(system=_SCREENSPOT_SYSTEM, element=element_name)
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data