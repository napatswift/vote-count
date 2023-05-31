import argparse
import os

from tqdm import trange

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('output_dir', type=str, default='vc-dataset')
    argparser.add_argument('num', type=int,)

    args = argparser.parse_args()

    from helpers import MMOCRFileSaver, generate_images, TextTemplate

    output_dir = os.path.join('output', args.output_dir)

    file_saver = MMOCRFileSaver(os.path.join(output_dir, 'imgs'),
                                os.path.join(output_dir))

    template_generator = TextTemplate()
    try:
        for x in trange(args.num):
            generate_images(file_saver, template_generator)
    except KeyboardInterrupt:
        pass

    file_saver.clean(os.path.join(output_dir, 'train.json'))
