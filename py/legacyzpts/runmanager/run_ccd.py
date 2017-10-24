from argparse import ArgumentParser

from legacyzpts.legacy_zeropoints import main,get_parser

def reproduce_error(camera,imagefn,ccdname):
    """run legacyzpts on a specific ccdname for imagefn

    Note: to reproduce an error in log files
    """
    cmd_line=['--camera', camera,'--image',imagefn,
            '--choose_ccd',ccdname]
    parser= get_parser()
    args = parser.parse_args(args=cmd_line)
    main(image_list=[args.image], args=args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--image',action='store',default=None,help='image filename',required=False)
    parser.add_argument('--ccdname',action='store',default=None,help='ccd to run',required=False)
    args = parser.parse_args()

    reproduce_error(args.camera,args.image,args.ccdname)
