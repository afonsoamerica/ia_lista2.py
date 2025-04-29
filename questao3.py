from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Criar modelo
modelo = DiscreteBayesianNetwork([
    ('Fumante', 'Infarto'),
    ('Obeso', 'Infarto'),
    ('Infarto', 'DorNoPeito'),
    ('Infarto', 'ECG_Anormal')
])

# Definir probabilidades condicionais
cpd_fumante = TabularCPD(
    variable='Fumante', variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'Fumante': ['Não', 'Sim']}
)

cpd_obeso = TabularCPD(
    variable='Obeso', variable_card=2,
    values=[[0.6], [0.4]],
    state_names={'Obeso': ['Não', 'Sim']}
)

cpd_infarto = TabularCPD(
    variable='Infarto', variable_card=2,
    values=[
        [0.98, 0.90, 0.85, 0.50],
        [0.02, 0.10, 0.15, 0.50]
    ],
    evidence=['Fumante', 'Obeso'],
    evidence_card=[2, 2],
    state_names={
        'Infarto': ['Não', 'Sim'],
        'Fumante': ['Não', 'Sim'],
        'Obeso': ['Não', 'Sim']
    }
)

cpd_dor = TabularCPD(
    variable='DorNoPeito', variable_card=2,
    values=[
        [0.5, 0.1],
        [0.5, 0.9]
    ],
    evidence=['Infarto'],
    evidence_card=[2],
    state_names={
        'DorNoPeito': ['Não', 'Sim'],
        'Infarto': ['Não', 'Sim']
    }
)

cpd_ecg = TabularCPD(
    variable='ECG_Anormal', variable_card=2,
    values=[
        [0.95, 0.05],
        [0.05, 0.95]
    ],
    evidence=['Infarto'],
    evidence_card=[2],
    state_names={
        'ECG_Anormal': ['Normal', 'Anormal'],
        'Infarto': ['Não', 'Sim']
    }
)

# Adicionar ao modelo
modelo.add_cpds(cpd_fumante, cpd_obeso, cpd_infarto, cpd_dor, cpd_ecg)

# Verificar modelo
assert modelo.check_model()

# Criar inferência
inferencia = VariableElimination(modelo)

# Função para imprimir tabela formatada
def imprimir_tabela(resultado, titulo=""):
    variavel = list(resultado.variables)[0]
    print(f"\n{titulo}")
    print("+---------------+-------------------+")
    print(f"| {variavel:13}      |     phi({variavel})  |")
    print("+====================+=====================+")
    
    for i, estado in enumerate(resultado.state_names[variavel]):
        prob = resultado.values[i]
        print(f"| {variavel}({i}){' '*(10-len(str(i)))}| {prob:17.4f} |")
        print("+-------------------+--------------------+")

# Exemplos de consulta
print("RESULTADOS DO DIAGNÓSTICO DE INFARTO")
print("====================================")

# 1. Probabilidade geral
resultado_geral = inferencia.query(variables=['Infarto'])
imprimir_tabela(resultado_geral, "Probabilidade geral de infarto:")

# 2. Com dor no peito
resultado_dor = inferencia.query(variables=['Infarto'], evidence={'DorNoPeito': 'Sim'})
imprimir_tabela(resultado_dor, "Dado Dor no Peito = Sim:")

# 3. Com ECG anormal e obeso
resultado_ecg_obeso = inferencia.query(variables=['Infarto'], evidence={'ECG_Anormal': 'Anormal', 'Obeso': 'Sim'})
imprimir_tabela(resultado_ecg_obeso, "Dado ECG Anormal e Obeso = Sim:")

# 4. Com todos os sintomas
resultado_completo = inferencia.query(
    variables=['Infarto'], 
    evidence={
        'Fumante': 'Sim',
        'DorNoPeito': 'Sim',
        'ECG_Anormal': 'Anormal'
    }
)
imprimir_tabela(resultado_completo, "Dado Fumante = Sim, Dor = Sim, ECG Anormal:")
