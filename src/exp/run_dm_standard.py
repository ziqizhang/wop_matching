'''
    About using custom embeddings

    The param 'embeddings' is only a name for identifying the embeddings. DM only support a limited set of
    these names. And when a recognisable name is supplied, it will attempt to download it. This means if you want to use
    a custom embedding, you need to 'hack it' by renaming your model using one of the expected names, and keep the
    same format. As an example, when embedding=fasttext.wiki.vec, DM will look for:
    https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

    You can pre-download this and save it into: [embeddings_cache_path] (see the code below)
    Then upon running this method, it will create a .pt file based on this downloaded embedding, saved in the same
    folder above.

    At the same time, some cache data are saved in the 'path' parameter (your data folder).


    By default, the next time DM runs, it will try to use the cache. So if you change 'embeddings', it will not work
    because it will try to look for the cache. You need to delete the cache, or set 'check_cached_data=False' in order
    to re-run DM from a fresh start and use a different embedding model
'''

import deepmatcher as dm
import sys
import time

if __name__ == "__main__":

    #sys.argv[1] is home_dir that forms the root folder to access embeddings, and saved models
    home_dir = sys.argv[1]

    #sys.argv[2] should be a relative path pointing to the folder where the embedding model used by DM is placed.
    #We will always pass the fasttext.wiki.vec alias as the embedding name, so make sure
    #inside this folder there is a file called wiki-news-300d-1M.vec
    embedding_cache_dir=home_dir+sys.argv[2]

    #sys.argv[3] should be a relative path pointing to the folder containing input data. DM will look for
    #three files: train.csv, validation.csv, test.csv. They must all be formatted in the required DM format
    data_dir = home_dir+sys.argv[3]

    #sys.argv[4] should be a relative path pointing to the output folder
    ourput_dir=home_dir+sys.argv[4]



    # train, validation, test = \
    #     dm.data.process(path=data_dir,
    #                     embeddings='fasttext.wiki.vec',
    #                     train='train.csv', validation='validation.csv', test='test.csv')



    train, validation, test = \
        dm.data.process(path=data_dir,
                        check_cached_data=False,
                        embeddings='fasttext.wiki.vec',
                        embeddings_cache_path=embedding_cache_dir,
                        train='train.csv', validation='validation.csv', test='test.csv')

    # parameters to keep consistent with
    nn_type = 'rnn'
    comp_type = 'abs-diff'
    #epochs = 15
    pos_neg_ratio = 1
    batch_size = 8
    lr = 0.001
    lr_decay = 0.9
    smoothing=0.05
    model = dm.MatchingModel(attr_summarizer=nn_type, attr_comparator=comp_type)
    model.initialize(train)
    optim = dm.optim.Optimizer(method='adam', lr=lr, max_grad_norm=5, start_decay_at=1, beta1=0.9, beta2=0.999, adagrad_accum=0.0, lr_decay=lr_decay)
    optim.set_parameters(model.named_parameters())
    start = time.time()
    model.run_train(
         train,
         validation,
         #epochs=epochs,
         batch_size=batch_size,
         pos_neg_ratio=pos_neg_ratio,
         optimizer=optim,
         label_smoothing=smoothing,
         best_save_path=ourput_dir+"/best_model.pth"
    )
    # end = time.time()
    # print('Training time: ' + str(end - start))
    # start = time.time()
    # model.run_eval(test, batch_size=batch_size)
    # end = time.time()
    # print('Prediction time: ' + str(end - start))

    #model = dm.MatchingModel()
    #model.run_train(train, validation, best_save_path=None)
    model.run_eval(test)

    # unlabeled = dm.data.process_unlabeled(path='data_directory/unlabeled.csv', trained_model=model)
    # model.run_prediction(unlabeled)