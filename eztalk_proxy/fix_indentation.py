with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

corrected_lines = [
    '                        \n', # Removed comment here
    '                        if request_data.provider == "openai":\n',
    '                            async for event in process_openai_response(parsed_sse_data, state, request_id):\n',
    '                                yield event\n',
    '                        elif request_data.provider == "google":\n',
    '                            async for event in process_google_response(parsed_sse_data, state, request_id):\n',
    '                                yield event\n',
]

lines[567:574] = corrected_lines

with open('main.py.fixed', 'w', encoding='utf-8') as f:
    f.writelines(lines)