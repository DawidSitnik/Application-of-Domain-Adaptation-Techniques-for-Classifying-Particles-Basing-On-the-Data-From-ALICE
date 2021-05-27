from domain_adaptation.adaptation.cdan import ParticlesClassifier as ParticlesClassifierCDAN
from domain_adaptation.adaptation.dan import ParticlesClassifier as ParticlesClassifierDAN
from domain_adaptation.adaptation.dann import ParticlesClassifier as ParticlesClassifierDANN
from domain_adaptation.adaptation.jan import ParticlesClassifier as ParticlesClassifierJAN
from domain_adaptation.adaptation.mdd import ParticlesClassifier as ParticlesClassifierMDD
from my_utils.config import Config


models_dict = {
        'source': {},
        'dan': {},
        'jan': {},
        'cdan': {},
        'dann': {},
        'mdd': {},
        'wdgrl': {}
    }

models_dict['jan']['model'] = ParticlesClassifierJAN
models_dict['jan']['fp'] = Config.jan_model_fp
models_dict['dan']['model'] = ParticlesClassifierDAN
models_dict['dan']['fp'] = Config.dan_model_fp
models_dict['cdan']['model'] = ParticlesClassifierCDAN
models_dict['cdan']['fp'] = Config.cdan_model_fp
models_dict['dann']['model'] = ParticlesClassifierDANN
models_dict['dann']['fp'] = Config.dann_model_fp
models_dict['mdd']['model'] = ParticlesClassifierMDD
models_dict['mdd']['fp'] = Config.mdd_model_fp
