import random
import csv
import argparse
from datetime import datetime, timedelta


class ArgsManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='StopSkipping-RLSAC')
        self.args = {}
        self.post_process = []
        self.vartypes = {}
        self.primitive_values = None

        self.add('mode', type=str)
        self.add('--embedding_size', default=128, type=int)
        self.add('--n_nodes', default=15, type=int)
        self.add('--seed', default=9112, type=int)
        self.add('--n_heads', default=4, type=int)
        self.add('--n_encoder_layers', default=6, type=int)
        self.add('--num_workers', default=2, type=int)

        self.add('--max_steps_per_episode', default=200, type=int)
        self.add('--random_sampling_steps', default=20_000, type=int)
        self.add('--steps_before_updating', default=20_000, type=int)
        self.add('--update_interval', default=1, type=int)
        self.add('--update_print_interval', default=5, type=int)
        self.add('--save_interval', default=5_000, type=int)
        self.add('--batch_size', default=64, type=int)
        self.add('--gamma', default=0.99, type=float)
        self.add('--tau', default=0.005, type=float)
        self.add('--lr', default=0.0003, type=float)
        self.add('--max_grad_norm', default=5.0, type=float)
        self.add('--device', default='cuda', type=str)
        self.add('--distinct_qnet_encoders', action='store_true')
        self.add('--var_tt', action='store_true')
        self.add('--transfer', default='', type=str)

        self.add('-v', '--version', default=0, type=int)

        self.add('--rl', action='store_true')
        self.add('--rl_evaluate', action='store_true')
        self.add('--sa', action='store_true')
        self.add('--ga', action='store_true')
        self.add('--ts', action='store_true')
        self.add('--n_evals', default=15, type=int)
        self.add('--eval_budget', default=30, type=int)


        self.add('--reward_done', default=0.0, type=float)
        self.add('--reward_step', default=-0.0001, type=float)
        self.add('--reward_step_nonaction', default=-1, type=float)
        self.add('--n_attempts_per_problem', default=1, type=int)

        self.add('--problem', default='', type=str)

        self.add('--disable_logging', action='store_true')

        def invert(x, args):
            return not x

        def identifier(x, args):
            if args['mode'] == 'train':
                curtime = datetime.now() + timedelta(hours=8)
                return curtime.strftime('%y%m%d_%H%M_') + generate_id()
            elif len(x) > 1:
                return x
            else:
                return x[0]

        self.add('file', nargs='*', default='', type=str,
                 rename='identifier', process_fn=identifier)

    def parse_value(self, val):
        if val == 'False':
            return False

        if val == 'True':
            return True

        try:
            val = float(val)
        except ValueError:
            return val

        val_int = int(val)

        if val != val_int:
            return val

        return val_int

    def load_instance(self, name):
        instances = []
        with open('run_params.csv', 'r') as f:
            instances.extend(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE))

        header = instances[0]
        idx_identifier = header.index('identifier')

        matched = [i + 1 for i, row in enumerate(instances[1:])
                   if name in row[idx_identifier]]

        if len(matched) > 1:
            matches = '\n'.join(['- ' + instances[x][idx_identifier] for x in matched])

            raise ValueError(
                "\n"
                "Multiple instances matched. Please select one of the following:\n"
                f"{matches}"
                "\n")
        elif len(matched) == 0:
            raise ValueError(
                "\n"
                "No matching instances found.\n")

        args_to_parse = {
            key: self.parse_value(val)
            for key, val in zip(header, instances[matched[0]])}

        identifier = args_to_parse['identifier']
        del args_to_parse['identifier']
        args = self.parse(args=args_to_parse)
        args['identifier'] = identifier

        return args, identifier

    def add(self, *name, target=None, process_fn=None, rename=None, **kwargs):
        pp = {}

        if 'out_type' in kwargs:
            vartype = kwargs['out_type']
            del kwargs['out_type']
        elif 'type' in kwargs:
            vartype = kwargs['type']
        elif 'action' in kwargs and kwargs['action'] in ['store_true', 'store_false']:
            vartype = bool

        if process_fn is not None:
            pp['process_fn'] = process_fn

        if rename is not None:
            pp['rename'] = rename
            varname = rename
        elif 'name' in kwargs:
            pp['name'] = kwargs['name']
            varname = kwargs['name']

            del kwargs['name']
        else:
            varname = self._trim(name[-1])

        self.vartypes[varname] = vartype

        if len(name) != 0:
            arg = self.parser.add_argument(*name, **kwargs)

        if len(pp) >= 1:
            pp['target'] = target if target is not None else arg.dest
            self.post_process.append(pp)

    def _trim(self, x):
        while x[0] == '-':
            x = x[1:]

        return x

    def parse(self, args=None):
        if args is None:
            args = self.parser.parse_args()
            args = args.__dict__

        self.primitive_values = args.copy()

        for opt in self.post_process:
            value = args[opt['target']]
            if 'process_fn' in opt:
                value = opt['process_fn'](value, args)

            if 'rename' in opt:
                args[opt['rename']] = value
                del args[opt['target']]
            elif 'name' in opt:
                args[opt['name']] = value
            else:
                args[opt['target']] = value
                # raise ValueError

        self.primitive_values['identifier'] = args['identifier']
        self.args = args
        return args

    def save_params(self):
        print(self.args['identifier'])

        params = sorted(self.primitive_values.items())

        rows = []
        with open('run_params.csv', 'r') as f:
            rows.extend(csv.reader(f))

        headers = rows[0]
        existing_fields = {x: v for x, v in params if x in headers}
        existing_fields = [existing_fields[x] for x in headers]

        if len(params) == len(headers):
            with open('run_params.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(existing_fields)
        else:
            new_headers, new_fields = map(list, zip(*[(x, v) for x, v in params if x not in headers]))

            with open('run_params.csv', 'w+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers + new_headers)

                for row in rows[1:]:
                    writer.writerow(row)

                writer.writerow(existing_fields + new_fields)

def generate_id():
    adjectives = ['abandoned', 'able', 'absolute', 'adorable', 'adventurous', 'academic', 'acceptable', 'acclaimed', 'accomplished', 'accurate', 'aching', 'acidic', 'acrobatic', 'active', 'actual', 'adept', 'admirable', 'admired', 'adolescent', 'adorable', 'adored', 'advanced', 'afraid', 'affectionate', 'aged', 'aggravating', 'aggressive', 'agile', 'agitated', 'agonizing', 'agreeable', 'ajar', 'alarmed', 'alarming', 'alert', 'alienated', 'alive', 'all', 'altruistic', 'amazing', 'ambitious', 'ample', 'amused', 'amusing', 'anchored', 'ancient', 'angelic', 'angry', 'anguished', 'animated', 'annual', 'another', 'antique', 'anxious', 'any', 'apprehensive', 'appropriate', 'apt', 'arctic', 'arid', 'aromatic', 'artistic', 'ashamed', 'assured', 'astonishing', 'athletic', 'attached', 'attentive', 'attractive', 'austere', 'authentic', 'authorized', 'automatic', 'avaricious', 'average', 'aware', 'awesome', 'awful', 'awkward', 'babyish', 'bad', 'back', 'baggy', 'bare', 'barren', 'basic', 'beautiful', 'belated', 'beloved', 'beneficial', 'better', 'best', 'bewitched', 'big', 'biodegradable', 'bitter', 'black', 'bland', 'blank', 'blaring', 'bleak', 'blind', 'blissful', 'blond', 'blue', 'blushing', 'bogus', 'boiling', 'bold', 'bony', 'boring', 'bossy', 'both', 'bouncy', 'bountiful', 'bowed', 'brave', 'breakable', 'brief', 'bright', 'brilliant', 'brisk', 'broken', 'bronze', 'brown', 'bruised', 'bubbly', 'bulky', 'bumpy', 'buoyant', 'burdensome', 'burly', 'bustling', 'busy', 'buttery', 'buzzing', 'calculating', 'calm', 'candid', 'canine', 'capital', 'carefree', 'careful', 'careless', 'caring', 'cautious', 'cavernous', 'celebrated', 'charming', 'cheap', 'cheerful', 'cheery', 'chief', 'chilly', 'chubby', 'circular', 'classic', 'clean', 'clear', 'clever', 'close', 'closed', 'cloudy', 'clueless', 'clumsy', 'cluttered', 'coarse', 'cold', 'colorful', 'colorless', 'colossal', 'comfortable', 'common', 'compassionate', 'competent', 'complete', 'complex', 'complicated', 'composed', 'concerned', 'concrete', 'confused', 'conscious', 'considerate', 'constant', 'content', 'conventional', 'cooked', 'cool', 'cooperative', 'coordinated', 'corny', 'corrupt', 'costly', 'courageous', 'courteous', 'crafty', 'crazy', 'creamy', 'creative', 'creepy', 'criminal', 'crisp', 'critical', 'crooked', 'crowded', 'cruel', 'crushing', 'cuddly', 'cultivated', 'cultured', 'cumbersome', 'curly', 'curvy', 'cute', 'cylindrical', 'damaged', 'damp', 'dangerous', 'dapper', 'daring', 'darling', 'dark', 'dazzling', 'dead', 'deadly', 'deafening', 'dear', 'dearest', 'decent', 'decimal', 'decisive', 'deep', 'defenseless', 'defensive', 'defiant', 'deficient', 'definite', 'definitive', 'delayed', 'delectable', 'delicious', 'delightful', 'delirious', 'demanding', 'dense', 'dental', 'dependable', 'dependent', 'descriptive', 'deserted', 'detailed', 'determined', 'devoted', 'different', 'difficult', 'digital', 'diligent', 'dim', 'dimpled', 'dimwitted', 'direct', 'disastrous', 'discrete', 'disfigured', 'disgusting', 'disloyal', 'dismal', 'distant', 'downright', 'dreary', 'dirty', 'disguised', 'dishonest', 'dismal', 'distant', 'distinct', 'distorted', 'dizzy', 'dopey', 'doting', 'double', 'downright', 'drab', 'drafty', 'dramatic', 'dreary', 'droopy', 'dry', 'dual', 'dull', 'dutiful', 'each', 'eager', 'earnest', 'early', 'easy', 'ecstatic', 'edible', 'educated', 'elaborate', 'elastic', 'elated', 'elderly', 'electric', 'elegant', 'elementary', 'elliptical', 'embarrassed', 'embellished', 'eminent', 'emotional', 'empty', 'enchanted', 'enchanting', 'energetic', 'enlightened', 'enormous', 'enraged', 'entire', 'envious', 'equal', 'equatorial', 'essential', 'esteemed', 'ethical', 'euphoric', 'even', 'evergreen', 'everlasting', 'every', 'evil', 'exalted', 'excellent', 'exemplary', 'exhausted', 'excitable', 'excited', 'exciting', 'exotic', 'expensive', 'experienced', 'expert', 'extraneous', 'extroverted', 'fabulous', 'failing', 'faint', 'fair', 'faithful', 'fake', 'false', 'familiar', 'famous', 'fancy', 'fantastic', 'far', 'faraway', 'fast', 'fat', 'fatal', 'fatherly', 'favorable', 'favorite', 'fearful', 'fearless', 'feisty', 'feline', 'female', 'feminine', 'few', 'fickle', 'filthy', 'fine', 'finished', 'firm', 'first', 'firsthand', 'fitting', 'fixed', 'flaky', 'flamboyant', 'flashy', 'flat', 'flawed', 'flawless', 'flickering', 'flimsy', 'flippant', 'flowery', 'fluffy', 'fluid', 'flustered', 'focused', 'fond', 'foolhardy', 'foolish', 'forceful', 'forked', 'formal', 'forsaken', 'forthright', 'fortunate', 'fragrant', 'frail', 'frank', 'frayed', 'free', 'french', 'fresh', 'frequent', 'friendly', 'frightened', 'frightening', 'frigid', 'frilly', 'frizzy', 'frivolous', 'front', 'frosty', 'frozen', 'frugal', 'fruitful', 'full', 'fumbling', 'functional', 'funny', 'fussy', 'fuzzy', 'gargantuan', 'gaseous', 'general', 'generous', 'gentle', 'genuine', 'giant', 'giddy', 'gigantic', 'gifted', 'giving', 'glamorous', 'glaring', 'glass', 'gleaming', 'gleeful', 'glistening', 'glittering', 'gloomy', 'glorious', 'glossy', 'glum', 'golden', 'good', 'gorgeous', 'graceful', 'gracious', 'grand', 'grandiose', 'granular', 'grateful', 'grave', 'gray', 'great', 'greedy', 'green', 'gregarious', 'grim', 'grimy', 'gripping', 'grizzled', 'gross', 'grotesque', 'grouchy', 'grounded', 'growing', 'growling', 'grown', 'grubby', 'gruesome', 'grumpy', 'guilty', 'gullible', 'gummy', 'hairy', 'half', 'handmade', 'handsome', 'handy', 'happy', 'hard', 'harmful', 'harmless', 'harmonious', 'harsh', 'hasty', 'hateful', 'haunting', 'healthy', 'heartfelt', 'hearty', 'heavenly', 'heavy', 'hefty', 'helpful', 'helpless', 'hidden', 'hideous', 'high', 'hilarious', 'hoarse', 'hollow', 'homely', 'honest', 'honorable', 'honored', 'hopeful', 'horrible', 'hospitable', 'hot', 'huge', 'humble', 'humiliating', 'humming', 'humongous', 'hungry', 'hurtful', 'husky', 'icky', 'icy', 'ideal', 'idealistic', 'identical', 'idle', 'idiotic', 'idolized', 'ignorant', 'ill', 'illegal', 'illiterate', 'illustrious', 'imaginary', 'imaginative', 'immaculate', 'immaterial', 'immediate', 'immense', 'impassioned', 'impeccable', 'impartial', 'imperfect', 'imperturbable', 'impish', 'impolite', 'important', 'impossible', 'impractical', 'impressionable', 'impressive', 'improbable', 'impure', 'inborn', 'incomparable', 'incompatible', 'incomplete', 'inconsequential', 'incredible', 'indelible', 'inexperienced', 'indolent', 'infamous', 'infantile', 'infatuated', 'inferior', 'infinite', 'informal', 'innocent', 'insecure', 'insidious', 'insignificant', 'insistent', 'instructive', 'insubstantial', 'intelligent', 'intent', 'intentional', 'interesting', 'internal', 'international', 'intrepid', 'ironclad', 'irresponsible', 'irritating', 'itchy', 'jaded', 'jagged', 'jaunty', 'jealous', 'jittery', 'joint', 'jolly', 'jovial', 'joyful', 'joyous', 'jubilant', 'judicious', 'juicy', 'jumbo', 'junior', 'jumpy', 'juvenile', 'kaleidoscopic', 'keen', 'key', 'kind', 'kindhearted', 'kindly', 'klutzy', 'knobby', 'knotty', 'knowledgeable', 'knowing', 'known', 'kooky', 'kosher', 'lame', 'lanky', 'large', 'last', 'lasting', 'late', 'lavish', 'lawful', 'lazy', 'leading', 'lean', 'leafy', 'left', 'legal', 'legitimate', 'light', 'lighthearted', 'likable', 'likely', 'limited', 'limp', 'limping', 'linear', 'lined', 'liquid', 'little', 'live', 'lively', 'livid', 'loathsome', 'lone', 'lonely', 'long', 'loose', 'lopsided', 'lost', 'loud', 'lovable', 'lovely', 'loving', 'low', 'loyal', 'lucky', 'lumbering', 'luminous', 'lumpy', 'lustrous', 'luxurious', 'mad', 'magnificent', 'majestic', 'major', 'male', 'mammoth', 'married', 'marvelous', 'masculine', 'massive', 'mature', 'meager', 'mealy', 'mean', 'measly', 'meaty', 'medical', 'mediocre', 'medium', 'meek', 'mellow', 'melodic', 'memorable', 'menacing', 'merry', 'messy', 'metallic', 'mild', 'milky', 'mindless', 'miniature', 'minor', 'minty', 'miserable', 'miserly', 'misguided', 'misty', 'mixed', 'modern', 'modest', 'moist', 'monstrous', 'monthly', 'monumental', 'moral', 'mortified', 'motherly', 'motionless', 'mountainous', 'muddy', 'muffled', 'multicolored', 'mundane', 'murky', 'mushy', 'musty', 'muted', 'mysterious', 'naive', 'narrow', 'nasty', 'natural', 'naughty', 'nautical', 'near', 'neat', 'necessary', 'needy', 'negative', 'neglected', 'negligible', 'neighboring', 'nervous', 'new', 'next', 'nice', 'nifty', 'nimble', 'nippy', 'nocturnal', 'noisy', 'nonstop', 'normal', 'notable', 'noted', 'noteworthy', 'novel', 'noxious', 'numb', 'nutritious', 'nutty', 'obedient', 'obese', 'oblong', 'oily', 'oblong', 'obvious', 'occasional', 'odd', 'oddball', 'offbeat', 'offensive', 'official', 'old', 'only', 'open', 'optimal', 'optimistic', 'opulent', 'orange', 'orderly', 'organic', 'ornate', 'ornery', 'ordinary', 'original', 'other', 'our', 'outlying', 'outgoing', 'outlandish', 'outrageous', 'outstanding', 'oval', 'overcooked', 'overdue', 'overjoyed', 'overlooked', 'palatable', 'pale', 'paltry', 'parallel', 'parched', 'partial', 'passionate', 'past', 'pastel', 'peaceful', 'peppery', 'perfect', 'perfumed', 'periodic', 'perky', 'personal', 'pertinent', 'pesky', 'pessimistic', 'petty', 'phony', 'physical', 'piercing', 'pink', 'pitiful', 'plain', 'plaintive', 'plastic', 'playful', 'pleasant', 'pleased', 'pleasing', 'plump', 'plush', 'polished', 'polite', 'political', 'pointed', 'pointless', 'poised', 'poor', 'popular', 'portly', 'posh', 'positive', 'possible', 'potable', 'powerful', 'powerless', 'practical', 'precious', 'present', 'prestigious', 'pretty', 'precious', 'previous', 'pricey', 'prickly', 'primary', 'prime', 'pristine', 'private', 'prize', 'probable', 'productive', 'profitable', 'profuse', 'proper', 'proud', 'prudent', 'punctual', 'pungent', 'puny', 'pure', 'purple', 'pushy', 'putrid', 'puzzled', 'puzzling', 'quaint', 'qualified', 'quarrelsome', 'quarterly', 'queasy', 'querulous', 'questionable', 'quick', 'quiet', 'quintessential', 'quirky', 'quixotic', 'quizzical', 'radiant', 'ragged', 'rapid', 'rare', 'rash', 'raw', 'recent', 'reckless', 'rectangular', 'ready', 'real', 'realistic', 'reasonable', 'red', 'reflecting', 'regal', 'regular', 'reliable', 'relieved', 'remarkable', 'remorseful', 'remote', 'repentant', 'required', 'respectful', 'responsible', 'repulsive', 'revolving', 'rewarding', 'rich', 'rigid', 'right', 'ringed', 'ripe', 'roasted', 'robust', 'rosy', 'rotating', 'rotten', 'rough', 'round', 'rowdy', 'royal', 'rubbery', 'rundown', 'ruddy', 'rude', 'runny', 'rural', 'rusty', 'sad', 'safe', 'salty', 'same', 'sandy', 'sane', 'sarcastic', 'sardonic', 'satisfied', 'scaly', 'scarce', 'scared', 'scary', 'scented', 'scholarly', 'scientific', 'scornful', 'scratchy', 'scrawny', 'second', 'secondary', 'secret', 'selfish', 'sentimental', 'separate', 'serene', 'serious', 'serpentine', 'several', 'severe', 'shabby', 'shadowy', 'shady', 'shallow', 'shameful', 'shameless', 'sharp', 'shimmering', 'shiny', 'shocked', 'shocking', 'shoddy', 'short', 'showy', 'shrill', 'shy', 'sick', 'silent', 'silky', 'silly', 'silver', 'similar', 'simple', 'simplistic', 'sinful', 'single', 'sizzling', 'skeletal', 'skinny', 'sleepy', 'slight', 'slim', 'slimy', 'slippery', 'slow', 'slushy', 'small', 'smart', 'smoggy', 'smooth', 'smug', 'snappy', 'snarling', 'sneaky', 'sniveling', 'snoopy', 'sociable', 'soft', 'soggy', 'solid', 'somber', 'some', 'spherical', 'sophisticated', 'sore', 'sorrowful', 'soulful', 'soupy', 'sour', 'spanish', 'sparkling', 'sparse', 'specific', 'spectacular', 'speedy', 'spicy', 'spiffy', 'spirited', 'spiteful', 'splendid', 'spotless', 'spotted', 'spry', 'square', 'squeaky', 'squiggly', 'stable', 'staid', 'stained', 'stale', 'standard', 'starchy', 'stark', 'starry', 'steep', 'sticky', 'stiff', 'stimulating', 'stingy', 'stormy', 'straight', 'strange', 'steel', 'strict', 'strident', 'striking', 'striped', 'strong', 'studious', 'stunning', 'stupendous', 'stupid', 'sturdy', 'stylish', 'subdued', 'submissive', 'substantial', 'subtle', 'suburban', 'sudden', 'sugary', 'sunny', 'super', 'superb', 'superficial', 'superior', 'supportive', 'surprised', 'suspicious', 'svelte', 'sweaty', 'sweet', 'sweltering', 'swift', 'sympathetic', 'tall', 'talkative', 'tame', 'tan', 'tangible', 'tart', 'tasty', 'tattered', 'taut', 'tedious', 'teeming', 'tempting', 'tender', 'tense', 'tepid', 'terrible', 'terrific', 'testy', 'thankful', 'that', 'these', 'thick', 'thin', 'third', 'thirsty', 'this', 'thorough', 'thorny', 'those', 'thoughtful', 'threadbare', 'thrifty', 'thunderous', 'tidy', 'tight', 'timely', 'tinted', 'tiny', 'tired', 'torn', 'total', 'tough', 'traumatic', 'treasured', 'tremendous', 'tragic', 'trained', 'tremendous', 'triangular', 'tricky', 'trifling', 'trim', 'trivial', 'troubled', 'true', 'trusting', 'trustworthy', 'trusty', 'truthful', 'tubby', 'turbulent', 'twin', 'ugly', 'ultimate', 'unacceptable', 'unaware', 'uncomfortable', 'uncommon', 'unconscious', 'understated', 'unequaled', 'uneven', 'unfinished', 'unfit', 'unfolded', 'unfortunate', 'unhappy', 'unhealthy', 'uniform', 'unimportant', 'unique', 'united', 'unkempt', 'unknown', 'unlawful', 'unlined', 'unlucky', 'unnatural', 'unpleasant', 'unrealistic', 'unripe', 'unruly', 'unselfish', 'unsightly', 'unsteady', 'unsung', 'untidy', 'untimely', 'untried', 'untrue', 'unused', 'unusual', 'unwelcome', 'unwieldy', 'unwilling', 'unwitting', 'unwritten', 'upbeat', 'upright', 'upset', 'urban', 'usable', 'used', 'useful', 'useless', 'utilized', 'utter', 'vacant', 'vague', 'vain', 'valid', 'valuable', 'vapid', 'variable', 'vast', 'velvety', 'venerated', 'vengeful', 'verifiable', 'vibrant', 'vicious', 'victorious', 'vigilant', 'vigorous', 'villainous', 'violet', 'violent', 'virtual', 'virtuous', 'visible', 'vital', 'vivacious', 'vivid', 'voluminous', 'wan', 'warlike', 'warm', 'warmhearted', 'warped', 'wary', 'wasteful', 'watchful', 'waterlogged', 'watery', 'wavy', 'wealthy', 'weak', 'weary', 'webbed', 'wee', 'weekly', 'weepy', 'weighty', 'weird', 'welcome', 'wet', 'which', 'whimsical', 'whirlwind', 'whispered', 'white', 'whole', 'whopping', 'wicked', 'wide', 'wiggly', 'wild', 'willing', 'wilted', 'winding', 'windy', 'winged', 'wiry', 'wise', 'witty', 'wobbly', 'woeful', 'wonderful', 'wooden', 'woozy', 'wordy', 'worldly', 'worn', 'worried', 'worrisome', 'worse', 'worst', 'worthless', 'worthwhile', 'worthy', 'wrathful', 'wretched', 'writhing', 'wrong', 'wry', 'yawning', 'yearly', 'yellow', 'yellowish', 'young', 'youthful', 'yummy', 'zany', 'zealous', 'zesty', 'zigzag']
    nouns =  ['aardvark', 'aardwolf', 'abalone', 'acaciarat', 'acouchi', 'addax', 'adder', 'adouri', 'aegeancat', 'agama', 'agouti', 'airedale', 'akitainu', 'albatross', 'albino', 'alleycat', 'alligator', 'allosaurus', 'alpaca', 'alpinegoat', 'ambushbug', 'ammonite', 'amoeba', 'amphibian', 'amphiuma', 'amurminnow', 'anaconda', 'anchovy', 'andeancat', 'anemone', 'angelfish', 'anglerfish', 'angora', 'angwantibo', 'anhinga', 'ankole', 'annelid', 'annelida', 'anole', 'antbear', 'anteater', 'antelope', 'antlion', 'anura', 'aoudad', 'apatosaur', 'aphid', 'appaloosa', 'aracari', 'arachnid', 'arawana', 'archerfish', 'arcticduck', 'arcticfox', 'arctichare', 'arcticseal', 'arcticwolf', 'argali', 'argusfish', 'arkshell', 'armadillo', 'armedcrab', 'armyant', 'armyworm', 'arrowana', 'arrowcrab', 'arrowworm', 'arthropods', 'aruanas', 'asianlion', 'astarte', 'atlasmoth', 'auklet', 'aurochs', 'avians', 'avocet', 'axisdeer', 'axolotl', 'ayeaye', 'aztecant', 'azurevase', 'babirusa', 'baboon', 'bactrian', 'badger', 'bagworm', 'baiji', 'baldeagle', 'ballpython', 'bandicoot', 'banteng', 'barasinga', 'barasingha', 'barbet', 'barnacle', 'barnowl', 'barracuda', 'basenji', 'basil', 'basilisk', 'beagle', 'beauceron', 'beaver', 'bedbug', 'beetle', 'bellfrog', 'bellsnake', 'betafish', 'bettong', 'bighorn', 'bilby', 'billygoat', 'binturong', 'bison', 'bittern', 'blackbear', 'blackbird', 'blackbuck', 'blackfish', 'blackfly', 'blacklab', 'blacklemur', 'blackmamba', 'blackrhino', 'blesbok', 'blobfish', 'blowfish', 'bluebird', 'bluebottle', 'bluefish', 'bluegill', 'bluejay', 'blueshark', 'bluet', 'bluewhale', 'bobcat', 'bobolink', 'bobwhite', 'boilweevil', 'bongo', 'bonobo', 'booby', 'borer', 'borzoi', 'boubou', 'boutu', 'bovine', 'brahmancow', 'brant', 'bream', 'bronco', 'brownbear', 'bubblefish', 'budgie', 'bufeo', 'buffalo', 'bufflehead', 'bullfrog', 'bumblebee', 'bunny', 'bunting', 'burro', 'bushbaby', 'bustard', 'butterfly', 'buzzard', 'caecilian', 'caiman', 'camel', 'canary', 'canine', 'canvasback', 'capybara', 'caracal', 'cardinal', 'caribou', 'cassowary', 'catbird', 'catfish', 'cattle', 'caudata', 'centipede', 'chafer', 'chameleon', 'chamois', 'cheetah', 'chevrotain', 'chick', 'chickadee', 'chicken', 'chihuahua', 'chimpanzee', 'chinchilla', 'chipmunk', 'chital', 'chrysalis', 'chuckwalla', 'chupacabra', 'cicada', 'cirriped', 'civet', 'clingfish', 'clumber', 'coati', 'cobra', 'cockatiel', 'cockatoo', 'cockroach', 'coelacanth', 'collie', 'comet', 'conch', 'condor', 'coney', 'conure', 'cooter', 'copepod', 'copperhead', 'coqui', 'coral', 'cormorant', 'cornsnake', 'cottontail', 'cougar', 'cowbird', 'cowrie', 'coyote', 'coypu', 'crane', 'cranefly', 'crayfish', 'creature', 'cricket', 'crocodile', 'crossbill', 'crustacean', 'cuckoo', 'curassow', 'curlew', 'cuscus', 'cusimanse', 'cuttlefish', 'cutworm', 'cygnet', 'dachshund', 'dairycow', 'dalmatian', 'damselfly', 'dartfrog', 'darwinsfox', 'dassie', 'dassierat', 'deermouse', 'degus', 'devilfish', 'dikdik', 'dikkops', 'dingo', 'dinosaur', 'diplodocus', 'dipper', 'discus', 'doctorfish', 'dodobird', 'dogfish', 'dolphin', 'donkey', 'dorado', 'dorking', 'dormouse', 'dotterel', 'dowitcher', 'drafthorse', 'dragon', 'dragonfly', 'drake', 'drever', 'dromedary', 'drongo', 'duckling', 'dugong', 'duiker', 'dungbeetle', 'dunlin', 'dunnart', 'eagle', 'earthworm', 'earwig', 'echidna', 'egret', 'eider', 'ekaltadeta', 'eland', 'elephant', 'elkhound', 'elver', 'equestrian', 'equine', 'ermine', 'eskimodog', 'fairyfly', 'falcon', 'fallowdeer', 'fantail', 'fanworms', 'feline', 'fennecfox', 'ferret', 'fieldmouse', 'finch', 'finwhale', 'fireant', 'firecrest', 'firefly', 'fishingcat', 'flamingo', 'flatfish', 'flicker', 'flies', 'flounder', 'fluke', 'flycatcher', 'flyingfish', 'flyingfox', 'fossa', 'foxhound', 'foxterrier', 'frogmouth', 'fruitbat', 'fruitfly', 'fulmar', 'furseal', 'gadwall', 'galago', 'galah', 'gallinule', 'gander', 'gannet', 'garpike', 'gavial', 'gazelle', 'gecko', 'geese', 'gelada', 'gelding', 'gemsbok', 'gemsbuck', 'genet', 'gerbil', 'gerenuk', 'gharial', 'gibbon', 'giraffe', 'glassfrog', 'globefish', 'glowworm', 'godwit', 'goitered', 'goldeneye', 'goldfinch', 'goldfish', 'gonolek', 'goose', 'goosefish', 'gopher', 'goral', 'gorilla', 'goshawk', 'gosling', 'gourami', 'grackle', 'grayfox', 'grayling', 'graywolf', 'greatargus', 'greatdane', 'grebe', 'grison', 'grosbeak', 'groundhog', 'grouper', 'grouse', 'grunion', 'guanaco', 'guillemot', 'guineafowl', 'guineapig', 'guppy', 'gypsymoth', 'gyrfalcon', 'hackee', 'haddock', 'hagfish', 'hairstreak', 'halcyon', 'halibut', 'halicore', 'hamadryad', 'hamadryas', 'hammerkop', 'hamster', 'hapuka', 'hapuku', 'harborseal', 'harpseal', 'harpyeagle', 'harrier', 'hartebeest', 'harvestmen', 'hedgehog', 'heifer', 'hellbender', 'herald', 'hermitcrab', 'heron', 'herring', 'hoatzin', 'hogget', 'hoiho', 'honeybee', 'honeyeater', 'hoopoe', 'hornbill', 'hornedtoad', 'hornet', 'hornshark', 'horse', 'horsefly', 'horsemouse', 'hound', 'housefly', 'hoverfly', 'huemul', 'human', 'husky', 'hydra', 'hyena', 'hyrax', 'ibisbill', 'icefish', 'ichidna', 'iggypops', 'iguana', 'iguanodon', 'illadopsis', 'imago', 'impala', 'incatern', 'inchworm', 'indianabat', 'indiancow', 'indianhare', 'indri', 'inganue', 'insect', 'isopod', 'ivorygull', 'izuthrush', 'jabiru', 'jackal', 'jackrabbit', 'jaeger', 'jaguar', 'jaguarundi', 'janenschia', 'javalina', 'jellyfish', 'jenny', 'jerboa', 'johndory', 'junco', 'junebug', 'kakapo', 'kakarikis', 'kangaroo', 'karakul', 'katydid', 'kawala', 'kestrel', 'killdeer', 'killifish', 'kingbird', 'kingfisher', 'kinglet', 'kingsnake', 'kinkajou', 'kiskadee', 'kissingbug', 'kitfox', 'kitten', 'kittiwake', 'kitty', 'koala', 'koalabear', 'kodiakbear', 'koodoo', 'kookaburra', 'kouprey', 'krill', 'kusimanse', 'lacewing', 'ladybird', 'ladybug', 'lamprey', 'langur', 'larva', 'lcont', 'leafbird', 'leafhopper', 'leafwing', 'leech', 'lemming', 'lemur', 'leonberger', 'leopard', 'leveret', 'lhasaapso', 'liger', 'limpet', 'limpkin', 'lionfish', 'lizard', 'llama', 'lobster', 'locust', 'longhorn', 'longspur', 'lorikeet', 'loris', 'louse', 'lovebird', 'lowchen', 'lunamoth', 'lungfish', 'macaque', 'macaw', 'macropod', 'maggot', 'magpie', 'maiasaura', 'malamute', 'mallard', 'maltesedog', 'mamba', 'mammal', 'mammoth', 'manatee', 'mandrill', 'mangabey', 'manta', 'mantaray', 'mantid', 'mantis', 'mantisray', 'manxcat', 'marabou', 'marlin', 'marmoset', 'marmot', 'marten', 'martin', 'massasauga', 'mastiff', 'mastodon', 'mayfly', 'meadowhawk', 'meadowlark', 'mealworm', 'meerkat', 'megaraptor', 'merganser', 'merlin', 'midge', 'milksnake', 'millipede', 'minibeast', 'minnow', 'mollies', 'mollusk', 'molly', 'monarch', 'mongoose', 'mongrel', 'monkey', 'moorhen', 'moose', 'moray', 'morayeel', 'morpho', 'mosasaur', 'mosquito', 'motmot', 'mouflon', 'mouse', 'mousebird', 'mudpuppy', 'mullet', 'muntjac', 'murrelet', 'muskox', 'muskrat', 'mussaurus', 'mussel', 'mustang', 'mynah', 'nabarlek', 'nagapies', 'nandine', 'nandoo', 'nandu', 'narwhal', 'narwhale', 'nauplius', 'nautilus', 'needlefish', 'needletail', 'nematode', 'neontetra', 'nerka', 'nettlefish', 'newtnutria', 'nighthawk', 'nightheron', 'nightjar', 'nilgai', 'noctilio', 'noctule', 'noddy', 'noolbenger', 'norwayrat', 'nubiangoat', 'nudibranch', 'numbat', 'nurseshark', 'nutcracker', 'nuthatch', 'nutria', 'nyala', 'nymph', 'ocelot', 'octopus', 'okapi', 'olingo', 'opossum', 'orangutan', 'oriole', 'oropendola', 'oropendula', 'osprey', 'ostracod', 'ostrich', 'otter', 'ovenbird', 'oxpecker', 'oyster', 'pachyderm', 'paddlefish', 'panda', 'pangolin', 'panther', 'paperwasp', 'papillon', 'parakeet', 'parrot', 'partridge', 'peacock', 'peafowl', 'peccary', 'pekingese', 'pelican', 'penguin', 'perch', 'pewee', 'phalarope', 'pheasant', 'phoebe', 'phoenix', 'pigeon', 'piglet', 'pilchard', 'pinemarten', 'pinniped', 'pintail', 'pipit', 'piranha', 'pitbull', 'pittabird', 'plankton', 'platypus', 'plover', 'polarbear', 'polecat', 'polliwog', 'polyp', 'pomeranian', 'pondskater', 'pooch', 'poodle', 'porcupine', 'porpoise', 'possum', 'prairiedog', 'prawn', 'primate', 'pronghorn', 'ptarmigan', 'pterosaurs', 'puffer', 'pufferfish', 'puffin', 'pullet', 'pupfish', 'puppy', 'pussycat', 'pygmy', 'python', 'quagga', 'quahog', 'quail', 'queenant', 'queenbee', 'queenconch', 'queensnake', 'quelea', 'quetzal', 'quillback', 'quokka', 'quoll', 'rabbit', 'raccoon', 'racer', 'racerunner', 'ragfish', 'raptors', 'rasbora', 'ratfish', 'rattail', 'raven', 'redhead', 'redpoll', 'redstart', 'reindeer', 'reptile', 'reynard', 'rhino', 'rhinoceros', 'ringworm', 'roach', 'roadrunner', 'robberfly', 'robin', 'rockrat', 'rodent', 'roebuck', 'roller', 'rooster', 'rottweiler', 'sable', 'saiga', 'sakimonkey', 'salamander', 'salmon', 'sambar', 'samoyeddog', 'sanddollar', 'sanderling', 'sandpiper', 'sapsucker', 'sardine', 'sawfish', 'scallop', 'scarab', 'scaup', 'schipperke', 'schnauzer', 'scorpion', 'scoter', 'screamer', 'seabird', 'seagull', 'seahog', 'seahorse', 'sealion', 'seamonkey', 'seaslug', 'seaurchin', 'seriema', 'serpent', 'serval', 'shark', 'shearwater', 'sheep', 'sheldrake', 'shelduck', 'shibainu', 'shihtzu', 'shorebird', 'shoveler', 'shrew', 'shrike', 'shrimp', 'siamang', 'siamesecat', 'sidewinder', 'sifaka', 'silkworm', 'silverfish', 'silverfox', 'siskin', 'skimmer', 'skink', 'skipper', 'skunk', 'skylark', 'sloth', 'slothbear', 'smelts', 'snail', 'snake', 'snipe', 'snowdog', 'snowgeese', 'snowmonkey', 'snowyowl', 'solenodon', 'solitaire', 'songbird', 'spadefoot', 'sparrow', 'sphinx', 'spider', 'spiketail', 'spittlebug', 'sponge', 'spoonbill', 'spreadwing', 'springbok', 'springtail', 'squab', 'squamata', 'squeaker', 'squid', 'squirrel', 'stagbeetle', 'stallion', 'starfish', 'starling', 'steed', 'steer', 'stilt', 'stingray', 'stinkbug', 'stinkpot', 'stoat', 'stonefly', 'stork', 'sturgeon', 'sunbear', 'sunbittern', 'sunfish', 'swallow', 'swellfish', 'swift', 'swordfish', 'tadpole', 'takin', 'tamarin', 'tanager', 'tapaculo', 'tapeworm', 'tapir', 'tarantula', 'tarpan', 'tarsier', 'taruca', 'tattler', 'tayra', 'tegus', 'teledu', 'tench', 'tenrec', 'termite', 'terrapin', 'terrier', 'thrasher', 'thrip', 'thrush', 'thylacine', 'tiger', 'tigermoth', 'tigershark', 'tilefish', 'tinamou', 'titmouse', 'toadfish', 'tortoise', 'toucan', 'towhee', 'tragopan', 'trogon', 'trout', 'tsetsefly', 'tuatara', 'turaco', 'turkey', 'turnstone', 'turtle', 'turtledove', 'uakari', 'ugandakob', 'umbrette', 'ungulate', 'unicorn', 'upupa', 'urchin', 'urial', 'urson', 'urubu', 'urutu', 'vampirebat', 'vaquita', 'veery', 'velvetcrab', 'velvetworm', 'verdin', 'vervet', 'vicuna', 'viper', 'viperfish', 'vipersquid', 'vireo', 'vixen', 'volvox', 'vulture', 'wallaby', 'wallaroo', 'walleye', 'walrus', 'warbler', 'warthog', 'waterbuck', 'waterbug', 'waterdogs', 'wattlebird', 'watussi', 'waxwing', 'weasel', 'weaverbird', 'weevil', 'whale', 'whapuku', 'whelp', 'whimbrel', 'whippet', 'whiteeye', 'whiterhino', 'whooper', 'widgeon', 'wildcat', 'wildebeast', 'wildebeest', 'willet', 'wireworm', 'wisent', 'wolfspider', 'wolverine', 'wombat', 'woodborer', 'woodchuck', 'woodcock', 'woodpecker', 'woodstorks', 'wrasse', 'wreckfish', 'wrenchbird', 'wryneck', 'wyvern', 'xanclomys', 'xanthareel', 'xantus', 'xenarthra', 'xenops', 'xenopus', 'xenurine', 'xerus', 'xiaosaurus', 'xiphias', 'xiphosuran', 'xrayfish', 'xraytetra', 'yaffle', 'yapok', 'yardant', 'yearling', 'yellowlegs', 'ynambu', 'yucker', 'zander', 'zebra', 'zebradove', 'zebrafinch', 'zebrafish', 'zenaida', 'zeren', 'zethuswasp', 'zopilote', 'zorilla']

    return random.choice(adjectives) + '_' + random.choice(nouns) + f'_{random.randint(0,99):02}'

if __name__ == "__main__":
    print(generate_id())