
from .PredFormer import PredFormer
from .benchmarks import CNNMethod, RNNMethod, LSTMMethod, ConvLSTMMethod

method_maps = {
    'predformer': PredFormer,
    'cnn': CNNMethod,
    'rnn': RNNMethod,
    'lstm': LSTMMethod,
    'convlstm': ConvLSTMMethod,
}

__all__ = [
    'method_maps', 'PredFormer', 'CNNMethod', 'RNNMethod', 'LSTMMethod', 'ConvLSTMMethod'
]
