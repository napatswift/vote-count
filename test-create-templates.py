import random
from faker import Faker
random.seed(0)


def generate_template():
    token_types = ['tambon', 'amphoe', 'province',
                'lastname_th', 'party_name_th',
                'name_th', 'title_th', 'number',
                'number_reading', 'number_th',
                'year', 'year_th', 'month_th',]
    
    def to_token_pattern(tokens):
        if isinstance(tokens, str):
            return '{'+tokens+'}'
        if isinstance(tokens, list):
            return '{'+('_'.join(tokens))+ '}'
    
    def get_random_token_type():
        token_type = random.choice(token_types)
        return to_token_pattern([token_type, 'handwriting'])
        
    total_tokens = random.randint(50, 500)
    faker: Faker = Faker('th_TH')
    tokens = []

    for i in range(total_tokens):
        if random.random() < 0.10:
            tokens.append(' ')
        elif random.random() < 0.008:
            tokens.append('\n')
            if random.random() < 0.5: tokens.append('\n')
            tokens.append(random.choice(['', '\t']))

        tokens.append(faker.word())

        if random.random() < 0.05:
            tokens.extend([' ',get_random_token_type(), ' '])
            if random.random() < 0.05:
                tokens.append(' ')
    
    return ''.join(tokens)

if __name__ == '__main__':
    for i in range(5):
        template = generate_template()
        with open('templates/gentemp_'+str(i)+'.txt', 'w') as fp:
            fp.write(template)