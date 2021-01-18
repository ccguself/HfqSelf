class Args(dict):
    def __init__(self, **kwargs):
        super(Args, self).__init__(**kwargs)

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = Args(value)
        return value
    
args_rzbl = Args(in_seq_length = 15,
            out_seq_length = 10,
            hidden_dim = 64,
            p_unit = 1,
            v_unit = 10000,
            t_unit = 2,
            o_unit = 1000,
            model_type = 0,
            is_tick = True,
            bar_s = 60,
            conv = [1,2,3,5],
              input_dim = 9,
              output_dim = 9,
              learning_rate = 0.0001,
              n_epochs= 100,
              batch_size = 32,
              prd_n =3)


args_cu = Args(in_seq_length = 15,
            out_seq_length = 10,
            hidden_dim = 128,
            p_unit = 10,
            v_unit = 1000000,
            t_unit = 1,
            model_type = 0,
            conv = [1,2,3,5],
              input_dim = 5,
              output_dim = 5,
              learning_rate = 0.0001,
              n_epochs= 100,
              batch_size = 32,
              prd_n =3)


args_btc = Args(in_seq_length = 15,
            out_seq_length = 10,
            hidden_dim = 128,
            p_unit = 1,
            v_unit = 1,
            t_unit = 1,
            model_type = 0,
            conv = [1,2,3,5],
              input_dim = 5,
              output_dim = 5,
              learning_rate = 0.0001,
              n_epochs= 100,
              batch_size = 32,
              prd_n =3)