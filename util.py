import os

def find_resume_json_weights(args):
    enclosing_dir = os.path.join(args.model_save, args.resume_subdir)

    files = os.listdir(enclosing_dir)

    model_jsons         = [f for f in files if (".json" in f and "hpe_hourglass_stacks" in f)]
    model_saved_weights = [f for f in files if (".hdf5" in f and "hpe_epoch{epoch:02d}".format(epoch=args.resume_epoch) in f)]
    
    assert len(model_jsons) > 0
    assert len(model_saved_weights) > 0

    args.resume_json = os.path.join(enclosing_dir, model_jsons[0])
    args.resume_weights = os.path.join(enclosing_dir, model_saved_weights[0])

    print('Found model json: {}'.format(args.resume_json))
    print('Found model weights for epoch {epoch:02d}: {weight_file_name}'.format(epoch=args.resume_epoch, weight_file_name=args.resume_weights))

    assert os.path.exists(args.resume_json)
    assert os.path.exists(args.resume_weights)