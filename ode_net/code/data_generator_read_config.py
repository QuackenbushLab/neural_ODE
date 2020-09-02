import configparser


def read_arguments_from_file(fp):
    """Reads run arguments from file"""
    config = configparser.ConfigParser()
    config.read(fp)
    settings = config['DEFAULT']
    return _convert_arguments(settings)

def _convert_arguments(settings):
    converted_settings = {}
    converted_settings['function'] = settings['function']
    converted_settings['filename_extension'] = settings['filename_extension']
    converted_settings['remove_factor'] = settings.getfloat('remove_factor')
    converted_settings['ntraj'] = settings.getint('ntraj')
    converted_settings['y0_range'] = tuple([int(e) for e in settings['y0_range'].split(",")])
    converted_settings['t_range'] = settings.getint('t_range')
    converted_settings['noise_scale'] = settings.getfloat('noise_scale')
    converted_settings['num_times'] = settings.getint('num_times')
    converted_settings['device'] = settings['device']
    converted_settings['random_y0'] = settings.getboolean('random_y0')
    return converted_settings
