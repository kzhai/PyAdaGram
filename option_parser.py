import optparse

delimiter = '-'


def floatable(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def intable(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def process_floats(option, opt_str, value, parser):
    assert value is None
    value = {}

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not floatable(arg):
            break

        tokens = arg.split("=")
        value[tokens[0]] = float(tokens[1])

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)

    return


def process_ints(option, opt_str, value, parser):
    assert value is None
    value = {}

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not int(arg):
            break

        tokens = arg.split("=")
        value[tokens[0]] = int(tokens[1])

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)

    return
