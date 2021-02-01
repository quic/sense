def clean_pipe_state_dict_key(key):
    to_replace = [
        ('feature_extractor', 'cnn'),
        ('feature_converter.', '')
    ]
    for pattern, replacement in to_replace:
        if key.startswith(pattern):
            key = key.replace(pattern, replacement)
    return key
