# Move this into train
import configparser


def read_arguments_from_file(fp):
    """Reads run arguments from file"""
    config = configparser.ConfigParser()
    config.read(fp)

    settings = config['settings']

    return _convert_arguments(settings)

def _convert_arguments(settings):
    converted_settings = {}
    converted_settings['viz'] = settings.getboolean('viz')
    converted_settings['viz_every_iteration'] = settings.getboolean('viz_every_iteration')
    converted_settings['verbose'] = settings.getboolean('verbose')
    converted_settings['method'] = settings['method']
    converted_settings['neurons_per_layer'] = settings.getint('neurons_per_layer')
    converted_settings['optimizer'] = settings['optimizer']

    converted_settings['batch_type'] = settings['batch_type']
    converted_settings['batch_size'] = settings.getint('batch_size')
    converted_settings['batch_time'] = settings.getint('batch_time')
    converted_settings['batch_time_frac'] = settings.getfloat('batch_time_frac')

    converted_settings['dec_lr'] = settings.getboolean('dec_lr')
    converted_settings['dec_lr_factor'] = settings.getfloat('dec_lr_factor')
    converted_settings['init_lr'] = settings.getfloat('init_lr')
    converted_settings['weight_decay'] = settings.getfloat('weight_decay')

    converted_settings['cpu'] = settings.getboolean('cpu')
    converted_settings['val_split'] = settings.getfloat('val_split')
    converted_settings['noise'] = settings.getfloat('noise')
    converted_settings['epochs'] = settings.getint('epochs')

    converted_settings['solve_eq_gridsize'] = settings.getint('solve_eq_gridsize')
    converted_settings['solve_A'] = settings.getboolean('solve_A')

    converted_settings['debug'] = settings.getboolean('debug')  
    converted_settings['output_dir'] = settings['output_dir']
    converted_settings['normalize_data'] = settings.getboolean('normalize_data')  
    converted_settings['explicit_time'] = settings.getboolean('explicit_time')
    converted_settings['relative_error'] = settings.getboolean('relative_error') 

    converted_settings['pretrained_model'] = settings.getboolean('pretrained_model') 
    
    return converted_settings
