import deepmatcher as dm
import numpy as np
np.random.seed(42)
import random
random.seed(42)

if __name__ == "__main__":
    data_dir = "/home/zz/Work/data/deepmatcher_toy/sample_data/itunes-amazon"

    train, validation, test = \
        dm.data.process(path=data_dir,
                        check_cached_data=False,
                        embeddings='fasttext.wiki.vec',
                        embeddings_cache_path=data_dir+"/embedding_cache",
                        train='train.csv', validation='validation.csv', test='test.csv')

    model = dm.MatchingModel()
    model.run_train(train, validation, best_save_path=None)
    model.run_eval(test)

    # unlabeled = dm.data.process_unlabeled(path='data_directory/unlabeled.csv', trained_model=model)
    # model.run_prediction(unlabeled)

'''
    This method is important for reading/caching embeddings

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