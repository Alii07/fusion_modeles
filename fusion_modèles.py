import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import io

st.title("Détection d'anomalies dans les cotisations URSSAF - Continental")

versement_mobilite = { '87005' : 1.80,  '87019': 1.80,  '87020': 1.80, '87038': 1.80, '87048': 1.80, '87050': 1.80, '87063': 1.80, '87065': 1.80, '87075': 1.80, '87085': 1.80, '87113': 1.80, '87114': 1.80, '87118': 1.80, '87125': 1.80,  '87143': 1.80, '87156': 1.80, '87192': 1.80,'87201': 1.80, '87202': 1.80,'87205': 1.80,'01027': 0.00, '01032': 0.00, '01049': 0.00, '01062': 0.00, '01142': 0.00, '01262': 0.00, '01276': 0.00, '01297': 0.00, '01342': 0.00, '08180': 1.00, '08199': 1.00, '08294': 1.00, '08040': 1.00, '08263': 1.00, '08298': 1.00, '08316': 1.00, '08327': 1.00, '08328': 1.00, '08331': 1.00, '08342': 1.00, '08343': 1.00, '08346': 1.00, '08371': 1.00, '08377': 1.00, '08385': 1.00, '08391': 1.00, '08400': 1.00, '08408': 1.00, '08409': 1.00, '08445': 1.00, '08457': 1.00, '08483': 1.00, '08475': 1.00, '08480': 1.00, '08481': 1.00, '08488': 1.00, '08491': 1.00, '08492': 1.00, '08494': 1.00, '08497': 1.00, '08003': 1.00, '08022': 1.00, '08043': 1.00, '08042': 1.00, '08053': 1.00, '08058': 1.00, '08072': 1.00, '08079': 1.00, '08096': 1.00, '08105': 1.00, '08114': 1.00, '08119': 1.00, '08125': 1.00, '08136': 1.00, '08137': 1.00, '08140': 1.00, '08142': 1.00, '08152': 1.00, '08158': 1.00, '08162': 1.00, '08170': 1.00, '08173': 1.00, '08174': 1.00, '08179': 1.00, '08187': 1.00, '08188': 1.00, '08191': 1.00, '08194': 1.00, '08209': 1.00, '08216': 1.00, '08230': 1.00, '08232': 1.00, '08235': 1.00, '08101': 1.00, '12145': 0.80, '12084': 0.80, '12002': 0.80, '12070': 0.80, '12072': 0.80, '12086': 0.80, '12204': 0.80, '48131': 0.80, '12160': 0.80, '12178': 0.80, '12180': 0.80, '12200': 0.80, '12211': 0.80, '12225': 0.80, '12293': 0.80, '14527': 0.80, '14069': 0.80, '14126': 0.80, '14141': 0.80, '14147': 0.80, '14177': 0.80, '14179': 0.80, '14193': 0.80, '14194': 0.80, '14260': 0.80, '14270': 0.80, '14293': 0.80, '14303': 0.80, '14326': 0.80, '14334': 0.80, '14082': 0.80, '14273': 0.80, '14337': 0.80, '14740': 0.80, '14419': 0.80, '14421': 0.80, '14425': 0.80, '14504': 0.80, '14520': 0.80, '14435': 0.80, '14362': 0.80, '14366': 0.80, '14368': 0.80, '14371': 0.80, '14403': 0.80, '14410': 0.80, '14431': 0.80, '14448': 0.80, '14460': 0.80, '14473': 0.80, '14474': 0.80, '14478': 0.80, '14484': 0.80, '14487': 0.80, '14522': 0.80, '14540': 0.80, '14571': 0.80, '14574': 0.80, '14582': 0.80, '14595': 0.80, '14621': 0.80, '14625': 0.80, '14626': 0.80, '14639': 0.80, '14648': 0.80, '14654': 0.80, '14576': 0.80, '14570': 0.80,'16010': 0.35, '16012': 0.35, '16013': 0.35, '16018': 0.35, '16032': 0.35, '16204': 0.35, '16045': 0.35, '16050': 0.35, '16056': 0.35, '16057': 0.35, '16058': 0.35, '16060': 0.35, '16077': 0.35, '16088': 0.35, '16089': 0.35, '16090': 0.35, '16097': 0.35, '16102': 0.35, '16116': 0.35, '16129': 0.35, '16139': 0.35, '16145': 0.35, '16150': 0.35, '16151': 0.35, '16152': 0.35, '16297': 0.35, '16163': 0.35, '16165': 0.35, '16167': 0.35, '16169': 0.35, '16171': 0.35, '16174': 0.35, '16186': 0.35, '16193': 0.35, '16153': 0.35, '16202': 0.35, '16216': 0.35, '16217': 0.35, '16218': 0.35, '16220': 0.35, '16233': 0.35, '16234': 0.35, '16243': 0.35, '16247': 0.35, '16277': 0.35, '16304': 0.35, '16316': 0.35, '16330': 0.35, '16340': 0.35, '16343': 0.35, '16349': 0.35, '16351': 0.35, '16352': 0.35, '16355': 0.35, '16359': 0.35, '16366': 0.35, '16369': 0.35, '16386': 0.35, '16097': 0.35, '16399': 0.35, '16402': 0.35, '16417': 0.35, '18006': 2.00, '18008': 2.00, '18028': 2.00, '18033': 2.00, '18050': 2.00, '18097': 2.00, '18129': 2.00, '18138': 2.00, '18157': 2.00, '18179': 2.00, '18180': 2.00, '18205': 2.00, '18207': 2.00, '18213': 2.00, '18218': 2.00, '18226': 2.00, '18255': 2.00, '18267': 2.00, '18288': 2.00, '18141': 2.00, '22002': 0.80, '22012': 0.80, '22015': 0.80, '22044': 0.80, '22053': 0.80, '22054': 0.80, '22076': 0.80, '22077': 0.80, '22079': 0.80, '22084': 0.80, '22093': 0.80, '22098': 0.80, '22114': 0.80, '22140': 0.80, '22151': 0.80, '22153': 0.80, '22154': 0.80, '22160': 0.80, '22165': 0.80, '22173': 0.80, '22175': 0.80, '22184': 0.80, '22185': 0.80, '22186': 0.80, '22193': 0.80, '22242': 0.80, '22246': 0.80, '22258': 0.80, '22261': 0.80, '22267': 0.80, '22273': 0.80, '22286': 0.80, '22296': 0.80, '22326': 0.80, '22332': 0.80, '22337': 0.80, '22341': 0.80, '22345': 0.80, '22346': 0.80, '22348': 0.80, '22369': 0.80, '24002': 1.60, '24010': 1.60, '24011': 1.60, '24013': 1.60, '24026': 1.60, '24044': 1.60, '24053': 1.60, '24061': 1.60, '24065': 1.60, '24092': 1.60, '24094': 1.60, '24098': 1.60, '24102': 1.60, '24103': 1.60, '24108': 1.60, '24115': 1.60, '24135': 1.60, '24138': 1.60, '24139': 1.60, '24146': 1.60, '24156': 1.60, '24160': 1.60, '24162': 1.60, '24166': 1.60, '24190': 1.60, '24208': 1.60, '24220': 1.60, '24239': 1.60, '24251': 1.60, '24256': 1.60, '24258': 1.60, '24266': 1.60, '24270': 1.60, '24312': 1.60, '24318': 1.60, '24322': 1.60, '24350': 1.60, '24362': 1.60, '24365': 1.60, '24369': 1.60, '24390': 1.60, '24421': 1.60, '24435': 1.60, '24439': 1.60, '24447': 1.60, '24459': 1.60, '24468': 1.60, '24480': 1.60, '24484': 1.60, '24518': 1.60, '24521': 1.60, '24527': 1.60, '24540': 1.60, '24557': 1.60, '24571': 1.60, '24576': 1.60, '25011': 1.80, '25020': 1.80, '25031': 1.80, '25040': 1.80, '25043': 1.80, '25048': 1.80, '25057': 1.80, '25097': 1.80, '25170': 1.80, '25188': 1.80, '25190': 1.80, '25196': 1.80, '25228': 1.80, '25230': 1.80, '25237': 1.80, '25284': 1.80, '25304': 1.80, '25367': 1.80, '25370': 1.80, '25388': 1.80, '25428': 1.80, '25526': 1.80, '25539': 1.80, '25547': 1.80, '25555': 1.80, '25580': 1.80, '25586': 1.80, '25614': 1.80, '25632': 1.80, '25004': 1.70, '25013': 1.70, '25033': 1.70, '25054': 1.70, '25059': 1.70, '25063': 1.70, '25071': 1.70, '25082': 1.70, '25093': 1.70, '25159': 1.70, '25187': 1.70, '25191': 1.70, '25192': 1.70, '25194': 1.70, '25207': 1.70, '25210': 1.70, '25214': 1.70, '25216': 1.70, '25224': 1.70, '25239': 1.70, '25274': 1.70, '25281': 1.70, '25316': 1.70, '25345': 1.70, '25350': 1.70, '25378': 1.70, '25394': 1.70, '25422': 1.70, '25426': 1.70, '25452': 1.70, '25463': 1.70, '25469': 1.70, '25481': 1.70, '25485': 1.70, '25497': 1.70, '25521': 1.70, '25523': 1.70, '25524': 1.70, '25540': 1.70, '25548': 1.70, '25562': 1.70, '25615': 1.70, '25617': 1.70, '25618': 1.70, '28013': 0.60, '28015': 0.60, '28023': 0.60, '28039': 0.60, '28058': 0.60, '28074': 0.60, '28092': 0.60, '28094': 0.60, '28113': 0.60, '28118': 0.60, '28135': 0.60, '28137': 0.60, '28140': 0.60, '28146': 0.60, '28168': 0.60, '28172': 0.60, '28188': 0.60, '28191': 0.60, '28207': 0.60, '28208': 0.60, '28213': 0.60, '28230': 0.60, '28249': 0.60, '28257': 0.60, '28268': 0.60, '28275': 0.60, '28279': 0.60, '28298': 0.60, '28299': 0.60, '28343': 0.60, '28349': 0.60, '28352': 0.60, '28357': 0.60, '28372': 0.60, '28379': 0.60, '28408': 0.60, '28417': 0.60, '28423': 0.60, '28425': 0.60, '29020': 1.25, '29048': 1.25, '29051': 1.25, '29066': 1.25, '29106': 1.25, '29107': 1.25, '29110': 1.25, '29134': 1.25, '29169': 1.25, '29170': 1.25, '29173': 1.25, '29216': 1.25, '29229': 1.25, '29232': 1.25, '29039': 0.70, '29049': 0.70, '29146': 0.70, '29153': 0.70, '29217': 0.70, '29241': 0.70, '29272': 0.70, '29281': 0.70, '29293': 0.70, '31003': 2.00, '31022': 2.00, '31025': 2.00, '31032': 2.00, '31035': 2.00, '31036': 2.00, '31004': 2.00, '31044': 2.00, '31048': 2.00, '31053': 2.00, '31056': 2.00, '31057': 2.00, '31058': 2.00, '31069': 2.00, '31088': 2.00, '31091': 2.00, '31113': 2.00, '31116': 2.00, '31117': 2.00, '31148': 2.00, '31149': 2.00, '31150': 2.00, '31151': 2.00, '31157': 2.00, '31161': 2.00, '31162': 2.00, '31163': 2.00, '31165': 2.00, '31169': 2.00, '31171': 2.00, '31181': 2.00, '31182': 2.00, '31184': 2.00, '31186': 2.00, '31187': 2.00, '31192': 2.00, '31203': 2.00, '31205': 2.00, '31227': 2.00, '31230': 2.00, '31240': 2.00, '31248': 2.00, '31253': 2.00, '31249': 2.00, '31254': 2.00, '31259': 2.00, '31273': 2.00, '31282': 2.00, '31284': 2.00, '31287': 2.00, '31293': 2.00, '31340': 2.00, '31351': 2.00, '31352': 2.00, '31355': 2.00, '31364': 2.00, '31366': 2.00, '31381': 2.00, '31384': 2.00, '31389': 2.00, '31395': 2.00, '31401': 2.00, '31402': 2.00, '31409': 2.00, '31410': 2.00, '31411': 2.00,'31417': 2.00, '31418': 2.00, '31420': 2.00, '31421': 2.00, '31424': 2.00, '31429': 2.00, '31433': 2.00, '31437': 2.00, '31445': 2.00, '31446': 2.00, '31448': 2.00, '31458': 2.00, '31460': 2.00, '31462': 2.00, '31467': 2.00, '31475': 2.00, '31484': 2.00, '31486': 2.00, '31488': 2.00, '31490': 2.00, '31497': 2.00, '31499': 2.00, '31506': 2.00, '31526': 2.00, '31533': 2.00, '31541': 2.00, '31547': 2.00, '31555': 2.00, '31557': 2.00, '31561': 2.00, '31568': 2.00, '31575': 2.00, '31578': 2.00, '31580': 2.00, '31588': 2.00, '31188': 2.00, '31277': 2.00, '31291': 2.00, '31297': 2.00, '31339': 2.00, '31496': 2.00, '35013': 0.55, '35045': 0.55, '35064': 0.55, '35145': 0.55, '35151': 0.55, '35219': 0.55, '35236': 0.55, '35237': 0.55, '35268': 0.55, '35285': 0.55, '35294': 0.55, '35328': 0.55, '044007': 0.55, '44044': 0.55, '44057': 0.55, '44067': 0.55, '44092': 0.55, '44123': 0.55, '44185': 0.55, '56001': 0.55, '56011': 0.55, '56060': 0.55, '56154': 0.55, '56194': 0.55, '56216': 0.55, '56221': 0.55, '56223': 0.55, '56232': 0.55, '56239': 0.55, '56250': 0.55, '41003': 0.60, '41138': 0.60, '41226': 0.60, '41269': 0.60, '41001': 0.35, '41004': 0.35, '41007': 0.35, '41010': 0.35, '41020': 0.35, '41030': 0.35, '41065': 0.35, '41070': 0.35, '41072': 0.35, '41073': 0.35, '41078': 0.35, '41079': 0.35, '41081': 0.35, '41087': 0.35, '41090': 0.35, '41098': 0.35, '41100': 0.35, '41102': 0.35, '41103': 0.35, '41107': 0.35, '41113': 0.35, '41120': 0.35, '41124': 0.35, '41131': 0.35, '41149': 0.35, '41153': 0.35, '41158': 0.35, '41163': 0.35, '41174': 0.35, '41182': 0.35, '41184': 0.35, '41186': 0.35, '41190': 0.35, '41192': 0.35, '41199': 0.35, '41200': 0.35, '41201': 0.35, '41209': 0.35, '41213': 0.35, '41215': 0.35, '41225': 0.35, '41228': 0.35, '41236': 0.35, '41238': 0.35, '41243': 0.35, '41250': 0.35, '41255': 0.35, '41259': 0.35, '41261': 0.35, '41263': 0.35, '41265': 0.35, '41274': 0.35, '41275': 0.35, '41278': 0.35, '41279': 0.35, '41283': 0.35, '41286': 0.35, '41287': 0.35, '41290': 0.35, '41291': 0.35, '41293': 0.35, '41294': 0.35, '44006': 0.80, '44010': 0.80, '44055': 0.80, '44049': 0.80, '44069': 0.80, '44072': 0.80, '44097': 0.80, '44125': 0.80, '44135': 0.80, '44175': 0.80, '44183': 0.80, '44211': 0.80, '56030': 0.80, '56058': 0.80, '56155': 0.80, '44012': 0.80, '44005': 0.80, '44038': 0.80, '44039': 0.80, '44106': 0.80, '44126': 0.80, '44131': 0.80, '44133': 0.80, '44136': 0.80, '44145': 0.80, '44164': 0.80, '44182': 0.80, '44186': 0.80, '44021': 0.80, '44220': 0.80, '46007': 0.80, '46032': 0.80, '46037': 0.80, '46040': 0.80, '46042': 0.80, '46044': 0.80, '46046': 0.80, '46064': 0.80, '46070': 0.80, '46077': 0.80, '46080': 0.80, '46088': 0.80, '46095': 0.80, '46109': 0.80, '46112': 0.80, '46119': 0.80, '46136': 0.80, '46137': 0.80, '46149': 0.80, '46156': 0.80, '46197': 0.80, '46134': 0.80, '46171': 0.80, '46188': 0.80, '46190': 0.80, '46191': 0.80, '46205': 0.80, '46211': 0.80, '46223': 0.80, '46224': 0.80, '46256': 0.80, '46264': 0.80, '46268': 0.80, '46280': 0.80, '46340': 0.80, '46320': 0.80, '46322': 0.80, '46327': 0.80, '46331': 0.80, '47006': 0.80, '47027': 0.80, '47049': 0.80, '47050': 0.80, '47053': 0.80, '47081': 0.80, '47099': 0.80, '47117': 0.80, '47075': 0.80, '47138': 0.80, '47146': 0.80, '47171': 0.80, '47215': 0.80, '47228': 0.80, '47239': 0.80, '47273': 0.80, '47237': 0.80, '47252': 0.80, '47323': 0.80, '50218': 0.55, '50165': 0.55, '50532': 0.55, '50647': 0.55, '50008': 0.40, '50038': 0.40, '50076': 0.40, '50081': 0.40, '50085': 0.40, '50102': 0.40, '50109': 0.40, '50117': 0.40, '50120': 0.40, '50143': 0.40, '50174': 0.40, '50188': 0.40, '50247': 0.40, '50252': 0.40, '50066': 0.40, '50237': 0.40, '50281': 0.40, '50327': 0.40, '50361': 0.40, '50278': 0.40, '50304': 0.40, '50277': 0.40, '50365': 0.40, '50447': 0.40, '50493': 0.40, '50540': 0.40, '50541': 0.40, '50549': 0.40, '51018': 0.60, '51029': 0.60, '51049': 0.60, '51093': 0.60, '51107': 0.60, '51110': 0.60, '51142': 0.60, '51154': 0.60, '51153': 0.60, '51196': 0.60, '51200': 0.60, '51202': 0.60, '51226': 0.60, '51230': 0.60, '51239': 0.60, '51251': 0.60, '51268': 0.60, '51271': 0.60, '51273': 0.60, '51281': 0.60, '51302': 0.60, '51327': 0.60, '51663': 0.60, '51342': 0.60, '51344': 0.60, '51367': 0.60, '51378': 0.60, '51384': 0.60, '51387': 0.60, '51390': 0.60, '51411': 0.60, '51413': 0.60, '51430': 0.60, '51431': 0.60, '51434': 0.60, '51435': 0.60, '51649': 0.60, '51499': 0.60, '51558': 0.60, '51578': 0.60, '51158': 0.60, '51603': 0.60, '51611': 0.60, '51612': 0.60, '51627': 0.60, '51630': 0.60, '51638': 0.60, '51643': 0.60, '51651': 0.60, '51655': 0.60, '53001': 1.00, '53007': 1.00, '53026': 1.00, '53034': 1.00, '53039': 1.00, '53040': 1.00, '53045': 1.00, '53049': 1.00, '53054': 1.00, '53056': 1.00, '53094': 1.00, '53099': 1.00, '53103': 1.00, '53108': 1.00, '53119': 1.00, '53129': 1.00, '53130': 1.00, '53137': 1.00, '53140': 1.00, '53141': 1.00, '53156': 1.00, '53157': 1.00, '53158': 1.00, '53168': 1.00, '53169': 1.00, '53175': 1.00, '53182': 1.00, '53201': 1.00, '53209': 1.00, '53224': 1.00, '53229': 1.00, '53410': 1.00, '53247': 1.00, '53262': 1.00, '57012': 1.80, '57022': 1.80, '57287': 1.80, '57067': 1.80, '57096': 1.80, '57124': 1.80, '57194': 1.80, '57199': 1.80, '57206': 1.80, '57221': 1.80, '57226': 1.80, '57242': 1.80, '57269': 1.80, '57305': 1.80, '57306': 1.80, '57323': 1.80, '57343': 1.80, '57356': 1.80, '57368': 1.80, '57372': 1.80, '57411': 1.80, '57441': 1.80, '57498': 1.80, '57508': 1.80, '57529': 1.80, '57562': 1.80, '57586': 1.80, '57647': 1.80, '57767': 1.80, '57666': 1.80, '57672': 1.80, '57678': 1.80, '57683': 1.80, '57731': 1.80, '57757': 1.80, '57013': 0.80, '57058': 0.80, '57101': 0.80, '57144': 0.80, '57176': 0.80, '57202': 0.80, '57208': 0.80, '57222': 0.80, '57227': 0.80, '57360': 0.80, '57466': 0.80, '57484': 0.80, '57514': 0.80, '57521': 0.80, '57537': 0.80, '57596': 0.80, '57638': 0.80, '57659': 0.80, '57660': 0.80, '57665': 0.80, '57669': 0.80, '58051': 0.80, '58088': 0.80, '58117': 0.80, '58121': 0.80, '58124': 0.80, '58126': 0.80, '58160': 0.80, '58194': 0.80, '58207': 0.80, '58214': 0.80, '58225': 0.80, '58278': 0.80, '58303': 0.80, '58238': 0.80, '69003': 2.00, '69029': 2.00, '69033': 2.00, '69034': 2.00, '69040': 2.00, '69044': 2.00, '69046': 2.00, '69271': 2.00, '69063': 2.00, '69273': 2.00, '69068': 2.00, '69069': 2.00, '69071': 2.00, '69072': 2.00, '69275': 2.00, '69081': 2.00, '69276': 2.00, '69085': 2.00, '69087': 2.00, '69088': 2.00, '69089': 2.00, '69278': 2.00, '69091': 2.00, '69096': 2.00, '69100': 2.00, '69279': 2.00, '69142': 2.00, '69250': 2.00, '69116': 2.00, '69117': 2.00, '69123': 2.00, '69381': 2.00, '69382': 2.00, '69383': 2.00, '69384': 2.00, '69385': 2.00, '69386': 2.00, '69387': 2.00, '69388': 2.00, '69389': 2.00, '69127': 2.00, '69282': 2.00, '69283': 2.00, '69284': 2.00, '69143': 2.00, '69149': 2.00, '69152': 2.00, '69153': 2.00, '69163': 2.00, '69286': 2.00, '69168': 2.00, '69191': 2.00, '69194': 2.00, '69199': 2.00, '69204': 2.00, '69205': 2.00, '69207': 2.00, '69290': 2.00, '69233': 2.00, '69202': 2.00, '69292': 2.00, '69293': 2.00, '69296': 2.00, '69244': 2.00, '69256': 2.00, '69259': 2.00, '69260': 2.00, '69266': 2.00, '69299': 2.00, '69277': 2.00, '69280': 2.00, '69285': 2.00, '69287': 2.00, '69288': 2.00, '69289': 2.00, '69298': 2.00, '69027': 1.15, '69043': 1.15, '69133': 1.15, '69136': 1.15, '69268': 1.15, '69013': 0.90, '69023': 0.90, '69061': 0.90, '69074': 0.90, '69092': 0.90, '01194': 0.90, '69105': 0.90, '69151': 0.90, '69115': 0.90, '69137': 0.90, '69167': 0.90, '69192': 0.90, '69197': 0.90, '69215': 0.90, '69172': 0.90, '69257': 0.90, '69264': 0.90, '69265': 0.90, '69270': 0.90, '69272': 0.90, '69281': 0.90, '69291': 0.90, '69294': 0.90, '69295': 0.90, '69297': 0.90, '69004': 0.80, '69005': 0.80, '69009': 0.80, '69017': 0.80, '69020': 0.80, '69026': 0.80, '69039': 0.80, '69047': 0.80, '69049': 0.80, '69050': 0.80, '69052': 0.80, '69055': 0.80, '69056': 0.80, '69059': 0.80, '69090': 0.80, '69106': 0.80, '69111': 0.80, '69113': 0.80, '69121': 0.80, '69122': 0.80, '69125': 0.80, '69126': 0.80, '69134': 0.80, '69140': 0.80, '69156': 0.80, '69159': 0.80, '69212': 0.80, '69239': 0.80, '69245': 0.80, '69246': 0.80, '69024': 0.80, '69021': 0.80, '69022': 0.80, '69032': 0.80, '69057': 0.80, '69067': 0.80, '69076': 0.80, '69083': 0.80, '69086': 0.80, '69010': 0.80, '69112': 0.80, '69171': 0.80, '69208': 0.80, '69216': 0.80, '69225': 0.80, '69229': 0.80, '69234': 0.80, '69240': 0.80, '69243': 0.80, '69248': 0.80, '69254': 0.80, '69157': 0.65, '69014': 0.65, '69030': 0.65, '69031': 0.65, '69038': 0.65, '42055': 0.65, '42062': 0.65, '69062': 0.65, '69078': 0.65, "42102":0.65, "69095":0.65, "69099":0.65, "69042":0.65, "69110":0.65, "69098":0.65, "69120":0.65, "42138":0.65, "69132":0.65, "69138":0.65, "69139":0.65, "69155":0.65, "69187":0.65, "42216":0.65, "69203":0.65, "69220":0.65, "69227":0.65, "69238":0.65, "69184":0.65, "69201":0.65, "69178":0.65, "69263":0.65, "42335":0.65, "42336":0.65, "69002":0.65, "69016":0.65, "69018":0.65, "69019":0.65, "69035":0.65, "69036":0.65, "69045":0.65, "69058":0.65, "69053":0.65, "69065":0.65, "69135":0.65, "69077":0.65, "69082":0.65, "69084":0.65, "69103":0.65, "69104":0.65, "69108":0.65, "69109":0.65, "69012":0.65, "69124":0.65, "69145":0.65, "69161":0.65, "69162":0.65, "69165":0.65, "69182":0.65, "69186":0.65, "69196":0.65, "69206":0.65, "69209":0.65, "69218":0.65, "69198":0.65, "69242":0.65, "69258":0.65, "69261":0.65, "69267":0.65, "73004":2.00, "73005":2.00, "73020":2.00, "73029":2.00, "73030":2.00, "73031":2.00, "73036":2.00, "73064":2.00, "73065":2.00, "73081":2.00, "73087":2.00, "73090":2.00, "73097":2.00, "73098":2.00,"73101":2.00,"73106":2.00,"73137":2.00,"73139":2.00,"73146":2.00,"73160":2.00,"73178":2.00,"73179":2.00,"73192":2.00,"73210":2.00,"73213":2.00,"73222":2.00,"73225":2.00,"73228":2.00,"73234":2.00,"73243":2.00,"73249":2.00,"73277":2.00,"73281":2.00,"73288":2.00,"73293":2.00,"73294":2.00,"73310":2.00,"73326":2.00,"73017":0.60,"73018":0.60,"73021":0.60,"73041":0.60,"73052":0.60,"73053":0.60,"73068":0.60,"73069":0.60,"73072":0.60,"73075":0.60,"73079":0.60,"73082":0.60,"73084":0.60,"73089":0.60,"73095":0.60,"73096":0.60,"73099":0.60,"73120":0.60,"73133":0.60,"73141":0.60,"73159":0.60,"73166":0.60,"73171":0.60,"73183":0.60,"73200":0.60,"73205":0.60,"73151":0.60,"73207":0.60,"73217":0.60,"73240":0.60,"73247":0.60,"73270":0.60,"73276":0.60,"73289":0.60,"73302":0.60,"73215":0.60,"73311":0.60,"73314":0.60,"73315":0.60,"73316":0.60,"73324":0.60,"80018":0.30,"80039":0.30,"76058":0.30,"80063":0.30,"80127":0.30,"80148":0.30,"76192":0.30,"80235":0.30,"80265":0.30,"76252":0.30,"76255":0.30,"76266":0.30,"80364":0.30,"80373":0.30,"76374":0.30,"76435":0.30,"76711":0.30,"76394":0.30,"76422":0.30,"80533":0.30,"76438":0.30,"76442":0.30,"80613":0.30,"76507":0.30,"76638":0.30,"80714":0.30,"76644":0.30,"80826":0.30,"79001":0.10,"79013":0.10,"79038":0.10,"79049":0.10,"79050":0.10,"79062":0.10,"79069":0.10,"79076":0.10,"79088":0.10,"79091":0.10,"79094":0.10,"79096":0.10,"79103":0.10,"79116":0.10,"79123":0.10,"79131":0.10,"79132":0.10,"79147":0.10,"79079":0.10,"79179":0.10,"79183":0.10,"79190":0.10,"79195":0.10,"79207":0.10,"79210":0.10,"79235":0.10,"79236":0.10,"79238":0.10,"79280":0.10,"79286":0.10,"79289":0.10,"79332":0.10,"79242":0.10,"87005":1.80,"87019":1.80,"87020":1.80,"87038":1.80,"87048":1.80,"87050":1.80,"87063":1.80,"87065":1.80,"87075":1.80,"87085":1.80,"87113":1.80,"87114":1.80,"87118":1.80,"87125":1.80,"87143":1.80,"87156":1.80,"87192":1.80,"87201":1.80,"87202":1.80,"87205":1.80,"88005":0.80,"88009":0.80,"88014":0.80,"88032":0.80,"88033":0.80,"88106":0.80,"88035":0.80,"88053":0.80,"88054":0.80,"88057":0.80,"88059":0.80,"88064":0.80,"88068":0.80,"88082":0.80,"88089":0.80,"88093":0.80,"88111":0.80,"88113":0.80,"88115":0.80,"88120":0.80,"88128":0.80,"88159":0.80,"88165":0.80,"88181":0.80,"88182":0.80,"88193":0.80,"88198":0.80,"88213":0.80,"88215":0.80,"88244":0.80,"88245":0.80,"88268":0.80,"88275":0.80,"88276":0.80,"88277":0.80,"88284":0.80,"88300":0.80,"88306":0.80,"88315":0.80,"88317":0.80,"88319":0.80,"88320":0.80,"88236":0.80,"88328":0.80,"88341":0.80,"88345":0.80,"88346":0.80,"88349":0.80,"88356":0.80,"88361":0.80,"88362":0.80,"88372":0.80,"88373":0.80,"88375":0.80,"88386":0.80,"88398":0.80,"88413":0.80,"88419":0.80,"88423":0.80,"88424":0.80,"88428":0.80,"88435":0.80,"88436":0.80,"88438":0.80,"88444":0.80,"88445":0.80,"88451":0.80,"88463":0.80,"88501":0.80,"88503":0.80,"88505":0.80,"88506":0.80,"88519":0.80,"88526":0.80,"54075":0.80,"54427":0.80,"54443":0.80,"28009":0.50,"28019":0.50,"28021":0.50,"28025":0.50,"28026":0.50,"28028":0.50,"28029":0.50,"28032":0.50,"28040":0.50,"28041":0.50,"28056":0.50,"28057":0.50,"28067":0.50,"28071":0.50,"28801":0.50,"28099":0.50,"28105":0.50,"28108":0.50,"28116":0.50,"28121":0.50,"28130":0.50,"28139":0.50,"28142":0.50,"28154":0.50,"28157":0.50,"28164":0.50,"28166":0.50,"28167":0.50,"28169":0.50,"28183":0.50,"28184":0.50,"28185":0.50,"28189":0.50,"28190":0.50,"28192":0.50,"28193":0.50,"28196":0.50,"28197":0.50,"28200":0.50,"28203":0.50,"28148":0.50,"28176":0.50,"28385":0.50,"28091":0.50,"28109":0.50,"28210":0.50,"28212":0.50,"28215":0.50,"28221":0.50,"28222":0.50,"28225":0.50,"28234":0.50,"28242":0.50,"28243":0.50,"28261":0.50,"28265":0.50,"28270":0.50,"28272":0.50,"28274":0.50,"28276":0.50,"28277":0.50,"28282":0.50,"28284":0.50,"28287":0.50,"28290":0.50,"28291":0.50,"28294":0.50,"28296":0.50,"28300":0.50,"28302":0.50,"28303":0.50,"28304":0.50,"28306":0.50,"28313":0.50,"28319":0.50,"28324":0.50,"28326":0.50,"28333":0.50,"28336":0.50,"28339":0.50,"28347":0.50,"28350":0.50,"28363":0.50,"28367":0.50,"28370":0.50,"28382":0.50,"28390":0.50,"28391":0.50,"28392":0.50,"28406":0.50,"28409":0.50,"28410":0.50,"28411":0.50,"28414":0.50,"28422":0.50,"28426":0.50,"28217":0.50,"28013":0.40,"28015":0.40,"28023":0.40,"28039":0.40,"28058":0.40,"28092":0.40,"28094":0.40,"28113":0.40,"28118":0.40,"28135":0.40,"28137":0.40,"28140":0.40,"28146":0.40,"28168":0.40,"28172":0.40,"28191":0.40,"28074":0.40,"28188":0.40,"28299":0.40,"28207":0.40,"28208":0.40,"28213":0.40,"28230":0.40,"28249":0.40,"28257":0.40,"28268":0.40,"28275":0.40,"28279":0.40,"28298":0.40,"28343":0.40,"28349":0.40,"28352":0.40,"28357":0.40,"28372":0.40,"28379":0.40,"28408":0.40,"28417":0.40,"28423":0.40,"28425":0.40,"34050":0.40,"34127":0.40,"34154":0.40,"34176":0.40,"34192":0.40,"34209":0.40,"34240":0.40,"34321":0.40,"34003":0.40,"34031":0.40,"34101":0.40,"34344":0.40,"34207":0.40,"34311":0.40,"34332":0.40,"34001":0.50,"34005":0.50,"34010":0.50,"34011":0.50,"34012":0.50,"34014":0.50,"34016":0.50,"34018":0.50,"34029":0.50,"34033":0.50,"34041":0.50,"34042":0.50,"34043":0.50,"34048":0.50,"34051":0.50,"34052":0.50,"34060":0.50,"34061":0.50,"34065":0.50,"34066":0.50,"34069":0.50,"34070":0.50,"34074":0.50,"34076":0.50,"34078":0.50,"34081":0.50,"34082":0.50,"34089":0.50,"34099":0.50,"34102":0.50,"34105":0.50,"34110":0.50,"34112":0.50,"34114":0.50,"34118":0.50,"34122":0.50,"34035":0.50,"34125":0.50,"34131":0.50,"34210":0.50,"34314":0.50,"34153":0.50,"34135":0.50,"34145":0.50,"34146":0.50,"34147":0.50,"34148":0.50,"34149":0.50,"34151":0.50,"34152":0.50,"34155":0.50,"34161":0.50,"34163":0.50,"34173":0.50,"34177":0.50,"34178":0.50,"34183":0.50,"34185":0.50,"34191":0.50,"34195":0.50,"34204":0.50,"34206":0.50,"34208":0.50,"34215":0.50,"34221":0.50,"34222":0.50,"34223":0.50,"34224":0.50,"34225":0.50,"34226":0.50,"34236":0.50,"34238":0.50,"34239":0.50,"34241":0.50,"34242":0.50,"34243":0.50,"34246":0.50,"34247":0.50,"34254":0.50,"34255":0.50,"34258":0.50,"34261":0.50,"34262":0.50,"34263":0.50,"34265":0.50,"34266":0.50,"34267":0.50,"34272":0.50,"34274":0.50,"34276":0.50,"34279":0.50,"34280":0.50,"34281":0.50,"34282":0.50,"34287":0.50,"34288":0.50,"34290":0.50,"34248":0.50,"34294":0.50,"34296":0.50,"34297":0.50,"34309":0.50,"34310":0.50,"34313":0.50,"34318":0.50,"34320":0.50,"34322":0.50,"34328":0.50,"34329":0.50,"34330":0.50,"34339":0.50,"34340":0.50,"34342":0.50,"34343":0.50,"34013":0.50,"34047":0.50,"34062":0.50,"34079":0.50,"34109":0.50,"34111":0.50,"34128":0.50,"34124":0.50,"34130":0.50,"34036":0.50,"34171":0.50,"34180":0.50,"34214":0.50,"34264":0.50,"34268":0.50,"34316":0.50,"34022":0.50,"34024":0.50,"34023":0.50,"34025":0.50,"34027":0.50,"34032":0.50,"34037":0.50,"34039":0.00,"34057":0.00,"34058":0.00,"34073":0.00,"34077":0.00,"34084":0.00,"34085":0.00,"34087":0.00,"34088":0.00,"34090":0.00,"34094":0.00,"34095":0.00,"34108":0.00,"34113":0.00,"34116":0.00,"34120":0.00,"34123":0.00,"34129":0.00,"34134":0.00,"34139":0.00,"34140":0.00,"34143":0.00,"34150":0.00,"34157":0.00,"34159":0.00,"34164":0.00,"34165":0.00,"34166":0.00,"34169":0.00,"34172":0.00,"34179":0.00,"34198":0.00,"34202":0.00,"34213":0.00,"34217":0.00,"34227":0.00,"34244":0.00,"34249":0.00,"34256":0.00,"34259":0.00,"34270":0.00,"34295":0.00,"34298":0.00,"34299":0.00,"34300":0.00,"34307":0.00,"34324":0.00,"34327":0.00,"34333":0.00,"34336":0.00,"34337":0.00,"34341":0.00,"34009":0.00,"34325":0.00,"74056":0.80,"74143":0.80,"74266":0.80,"74290":0.80,"07010":0.55,"07041":0.55,"07078":0.55,"07160":0.55,"07197":0.55,"07225":0.55,"07227":0.55,"07258":0.55,"07265":0.55,"07310":0.55,"07317":0.55,"07321":0.55,"07333":0.55,"07337":0.55,"07432":0.55,"07347":0.55 }

models_info = {
    'Mosselle': {
        'model': tf.keras.models.load_model('./Maladie/Moselle/maladie_Moselle.keras'),
        'numeric_cols': ['Brut_mensuel', 'Absences par Jour', 'Absences par Heure', 'Base Alsace Moselle',
                         'ASSIETTE CUM', 'SMIC M CUM', 'PLAFOND CUM', 'MALADIE CUM',
                         'MontantSal Alsace Moselle', 'Taux2 Alsace Moselle',
                         'Taux Alsace Moselle', 'MontantPat Alsace Moselle'],
        'categorical_cols': ['Region'],
        'target_col': 'Fraud Maladie Moselle'
    },
    '7030': {
        'model': tf.keras.models.load_model('./allocations/7030/allocations_7030.keras'),
        'numeric_cols': [ 'Absences par Jour', 'Absences par Heure', 'PLAFOND CUM',
       'ASSIETTE CUM', 'ALL FAM CUM', '7030Base', '7030Taux 2','7030Montant Pat.'],
                       'categorical_cols': [],
        'target_col': '7030 Fraud'
    },
    '7025': {
        'model': tf.keras.models.load_model('./allocations/7025/allocations_7025.keras'),
        'numeric_cols': [ 'Absences par Jour', 'Absences par Heure', 'PLAFOND CUM',
       'ASSIETTE CUM', 'ALL FAM CUM', '7025Base', '7025Taux 2','7025Montant Pat.'],
                       'categorical_cols': [],
        'target_col': '7025 Fraud'
    },
    '6000': {
        'model': tf.keras.models.load_model('./Maladie/6000/maladie_6000.keras'),
        'numeric_cols': ['Absences par Jour',
       'Absences par Heure',  'PLAFOND CUM',
       'ASSIETTE CUM', 'MALADIE CUM', '6000Base', '6000Taux',
       '6000Montant Sal.'],
         'categorical_cols': ['Catégorie texte'],
         'target_col': '6000 Fraud'
     },
     '6080': {
        'model': tf.keras.models.load_model('./CSG_CRDS/6080.keras'),
        'numeric_cols': ['Absences par Heure', '6080Base', '6080Taux', '6080Montant Sal.'],
         'categorical_cols': ['Statut texte'],
         'target_col': '6080 Fraud'
     },
     '6084': {
        'model': tf.keras.models.load_model('./CSG_CRDS/6084.keras'),
        'numeric_cols': ['Absences par Heure', '6084Base', '6084Taux', '6084Montant Sal.'],
        'categorical_cols': ['Statut texte'],
        'target_col': '6084 Fraud'
     },
    '7001': {
        'model': tf.keras.models.load_model('./Maladie/7001/maladie_7001.keras'),
        'numeric_cols': ['Matricule', 'Absences par Jour',
       'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CUM','MALADIE CUM',  '7001Base', '7001Taux 2',
       '7001Montant Pat.'],
        'categorical_cols': [],
        'target_col': '7001 Fraud'
    },
    '7002': {
        'model': tf.keras.models.load_model('./Maladie/7002/maladie_7002.keras'),
        'numeric_cols': ['Matricule', 'Absences par Jour',
       'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CUM',
       'MALADIE CUM',  '7002Base', '7002Taux 2', '7002Montant Pat.'],
        'categorical_cols': [],
        'target_col': '7002 Fraud'
    },
    '7010': {
            'model': tf.keras.models.load_model('./Reste/7010.keras'),
            'numeric_cols': ['7010Taux', '7010Taux 2', '7010Base'],
            'categorical_cols': [],
            'target_col': '7010 Fraud'
    },
    '7020': {
            'model': tf.keras.models.load_model('./FNAL/7020.keras'),
            'numeric_cols': ['Effectif', 'ASSIETTE CUM', 'PLAFOND CUM', '7020Taux', '7020Montant Pat.', 'Absences par Jour'],
            'categorical_cols': ['Statut texte'],
            'target_col': '7020 Fraud'
    },
    '7050': {
            'model': tf.keras.models.load_model('./Reste/7050.keras'),
            'numeric_cols': ['7050Base', '7050Taux 2', '7050Montant Pat.'],
            'categorical_cols': [],
            'target_col': '7050 Fraud'
    },
    '7035': {
            'model': tf.keras.models.load_model('./Reste/7035.keras'),
            'numeric_cols': ['7035Taux 2', '7035Montant Pat.', '7035Base'],
            'categorical_cols': [],
            'target_col': '7035 Fraud'
    },
    '7C00': {
            'model': tf.keras.models.load_model('./Reste/7C00.keras'),
            'numeric_cols': ['7C00Taux 2', '7C00Base', '7C00Montant Pat.'],
            'categorical_cols': [],
            'target_col': '7C00 Fraud'
    },
    '7040': {
            'model': tf.keras.models.load_model('./Reste/7040.keras'),
            'numeric_cols': ['7040Taux 2', '7040Base', '7040Montant Pat.'],
            'categorical_cols': [],
            'target_col': '7040 Fraud'
    },
    '7C20': {
            'model': tf.keras.models.load_model('./Reste/7C20.keras'),
            'numeric_cols': ['7C20Taux 2', '7C20Montant Pat.', '7C20Base'],
            'categorical_cols': [],
            'target_col': '7C20 Fraud'
    },

}

csv_upload = st.file_uploader("Entrez votre bulletin de paie (Format csv)", type=['csv'])

def apply_versement_conditions(df):
    required_columns = ['Versement Mobilite Taux', 'Effectif', 'Code Insee', 'Versement mobilite Base', 'versement Mobilite Montant Pat.']
    
    if not all(col in df.columns for col in required_columns):
        df['Mobilite_Anomalie'] = False
        df['mobilite Fraud'] = 0
        return df

    df['Versement Mobilite Taux Avant'] = df['Versement Mobilite Taux']
    df['Versement Mobilite Taux'] = df.apply(
        lambda row: 0 if row['Effectif'] < 11 else versement_mobilite.get(row['Code Insee'], row['Versement Mobilite Taux']),
        axis=1
    )

    df['Versement Mobilite Montant Pat. Calcule'] = (df['Versement mobilite Base'] * df['Versement Mobilite Taux'] / 100).round(2)
    df['Mobilite_Montant_Anomalie'] = df['versement Mobilite Montant Pat.'] != df['Versement Mobilite Montant Pat. Calcule']
    df['Mobilite_Anomalie'] = (df['Versement Mobilite Taux Avant'] != df['Versement Mobilite Taux']) | df['Mobilite_Montant_Anomalie']
    df['mobilite Fraud'] = 0
    df.loc[df['Mobilite_Anomalie'], 'mobilite Fraud'] = 1

    return df

def process_model(df, model_name, info, anomalies_report, model_anomalies):
    df_filtered = df

    if df_filtered.empty:
        st.write(f"Aucune donnée à traiter pour le modèle {model_name}.")
        return

    required_columns = info['numeric_cols'] + info['categorical_cols']

    if all(col in df_filtered.columns for col in required_columns):
        df_inputs = df_filtered.drop(columns=[info['target_col']]) if info['target_col'] in df_filtered.columns else df_filtered

        # Encodage des colonnes catégorielles
        if info['categorical_cols']:
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_categorical = one_hot_encoder.fit_transform(df_inputs[info['categorical_cols']])
        else:
            X_categorical = np.empty((len(df_inputs), 0))

        X_numeric = df_inputs[info['numeric_cols']].applymap(lambda x: str(x).replace(',', '.'))
        X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)

        X_test = np.concatenate([X_categorical, X_numeric_scaled], axis=1)

        expected_input_shape = info['model'].input_shape[-1]
        if X_test.shape[1] != expected_input_shape:
            st.error(f"Erreur : Le modèle {model_name} attend une entrée de forme {expected_input_shape} mais a reçu {X_test.shape[1]}.")
            return

        y_pred_proba = info['model'].predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype("int32")

        df.loc[df_filtered.index, f'{model_name}_Anomalie_Pred'] = y_pred

        num_anomalies = np.sum(y_pred)
        model_anomalies[model_name] = num_anomalies

        for index in df_filtered.index:
            if y_pred[df_filtered.index.get_loc(index)] == 1:
                anomalies_report.setdefault(index, set()).add(model_name)

def detect_anomalies(df):
    anomalies_report = {}
    model_anomalies = {}

    df = apply_versement_conditions(df)

    for model_name, info in models_info.items():
        process_model(df, model_name, info, anomalies_report, model_anomalies)

    st.write("**Rapport d'anomalies détectées :**")
    total_anomalies = len(anomalies_report)
    st.write(f"**Total des lignes avec des anomalies :** {total_anomalies}")

    report_content = io.StringIO()
    report_content.write("Rapport d'anomalies détectées :\n\n")
    report_content.write(f"Total des lignes avec des anomalies : {total_anomalies}\n")
    for model_name, count in model_anomalies.items():
        report_content.write(f"Le modèle {model_name} a détecté {count} anomalies.\n")

    for line_index, models in anomalies_report.items():
        report_content.write(f"Ligne {line_index + 1} : anomalie dans les cotisations {', '.join(sorted(models))}\n")

    report_content.seek(0)
    return report_content

def charger_dictionnaire(fichier):
    dictionnaire = {}
    try:
        with open(fichier, 'r', encoding='utf-8') as f:
            for ligne in f:
                code, description = ligne.strip().split(' : ', 1)
                dictionnaire[code] = description
    except FileNotFoundError:
        st.error(f"Le fichier {fichier} n'a pas été trouvé.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du dictionnaire : {e}")
    return dictionnaire

if csv_upload:
    dictionnaire = charger_dictionnaire('./Dictionnaire.txt')
    df_dictionnaire = pd.DataFrame(list(dictionnaire.items()), columns=['Code', 'Description'])

    st.header('Dictionnaire des Codes et Descriptions')
    st.dataframe(df_dictionnaire)

    try:
        df = pd.read_csv(csv_upload, encoding='utf-8', on_bad_lines='skip')
    except pd.errors.EmptyDataError:
        st.error("Le fichier CSV est vide ou mal formaté.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_upload, encoding='ISO-8859-1', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
    except Exception as e:
        st.error(f"Erreur inattendue lors de la lecture du fichier CSV : {e}")
    else:
        df.columns = df.columns.str.strip()
        report_content = detect_anomalies(df)
        st.download_button(
            label="Télécharger le rapport d'anomalies",
            data=report_content.getvalue(),
            file_name='anomalies_report.txt',
            mime='text/plain'
        )
