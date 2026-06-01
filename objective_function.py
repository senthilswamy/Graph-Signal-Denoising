import tensorflow as tf

from model_builder import build_model


class ObjectiveFunction:

    def __init__(
            self,
            X,
            A_hat,
            num_nodes):

        self.X = X
        self.A_hat = A_hat
        self.num_nodes = num_nodes

    def __call__(self, params):

        try:

            model = build_model(
                params,
                self.A_hat,
                self.num_nodes
            )

            history = model.fit(
                self.X,
                self.X,
                epochs=2,
                batch_size=int(params[1]),
                validation_split=0.2,
                verbose=1
            )

            loss = min(
                history.history['val_loss']
            )

            tf.keras.backend.clear_session()

            return loss

        except:

            return 999999