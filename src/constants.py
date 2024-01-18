ROOT = '/apdcephfs/share_916081/billmzhang/consistency'  # obsolute root directory for the project
interlocutor_tokens = ['[B]', '[A]']
CHECK_DIAG = 'check_diag'
CHECK_TURN = 'check_turn'
RESOLVE_TURN = 'resolve_turn'
RESOLVE_DIAG = 'resolve_diag'

CONTEXTUAL_CONSISTENCY_CHECKING_PER_TURN = 'contextual_checking_per_turn'

CNLI_LABEL_CONTRADICTION = 'contradiction'
CNLI_LABEL_NEUTRAL = 'neutral'
CNLI_LABEL_ENTAILMENT = 'entailment'

DATA_LOADING_SEQ = ['tcon', 'cdconv', 'stance', 'ocnli']
ID_2_LABEL = {0: 'consistent', 1: 'inconsistent'}
LABEL_2_ID = {'consistent': 0, 'inconsistent': 1}

from transformers import BartForConditionalGeneration, BertForSequenceClassification, T5ForConditionalGeneration
from modeling_cpt import CPTForConditionalGeneration
ARCH_MAPPING = {'bart': BartForConditionalGeneration,
                'bert': BertForSequenceClassification,
                'cpt': CPTForConditionalGeneration,
                't5': T5ForConditionalGeneration}
