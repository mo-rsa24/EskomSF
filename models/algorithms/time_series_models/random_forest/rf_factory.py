from models.algorithms.time_series_models.random_forest.utils import frequency_encode, target_encode


class EncoderFactory:
    """
    Factory to produce encoder functions: 'frequency' or 'target'.
    """
    @staticmethod
    def get_encoder(method: str):
        if method == "frequency":
            return frequency_encode
        elif method == "target":
            return target_encode
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
