import os
import random
import re
import attacut


class TextTemplate:
    def __init__(self) -> None:
        self.template_files = [os.path.join('templates', f) for f in os.listdir(
            'templates') if f.endswith('.txt')]
        random.shuffle(self.template_files)
        self.current_idx = 0

    def gen(self,):
        template_file = self.template_files[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.template_files)
        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()

        return self._get_tokens(template)

    def _get_tokens(self, text_template):
        big_pieces = re.split(
            '(\<[^\>]*\>|\t|\n| |\{[^\}]*\}|%%\w+|\|)', text_template)
        tokens = []

        for piece in big_pieces:
            if re.findall('[\u0E00-\u0E0F]+', piece):
                tokens.extend(attacut.tokenize(piece))
            elif piece == '':
                continue
            else:
                tokens.append(piece)
        return tokens
