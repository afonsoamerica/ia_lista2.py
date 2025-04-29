from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definir a estrutura da rede
modelo = DiscreteBayesianNetwork([
    ('Fumante', 'Infarto'),
    ('Obeso', 'Infarto'),
    ('Infarto', 'DorNoPeito'),
    ('Infarto', 'ECG_Anormal')
])

# 2. Definir as probabilidades condicionais
# Probabilidades a priori
cpd_fumante = TabularCPD(
    variable='Fumante',
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'Fumante': ['Não', 'Sim']}
)

cpd_obeso = TabularCPD(
    variable='Obeso',
    variable_card=2,
    values=[[0.6], [0.4]],
    state_names={'Obeso': ['Não', 'Sim']}
)

# Probabilidade de Infarto
# Probabilidade revisada de Infarto
cpd_infarto = TabularCPD(
    variable='Infarto',
    variable_card=2,
    values=[
        # Fumante Não, Obeso Não | Não, Sim | Sim, Não | Sim, Sim
        [0.98, 0.90, 0.85, 0.50],  # Infarto = Não (aumentei as probabilidades base)
        [0.02, 0.10, 0.15, 0.50]   # Infarto = Sim (risco mais realista)
    ],
    evidence=['Fumante', 'Obeso'],
    evidence_card=[2, 2],
    state_names={
        'Infarto': ['Não', 'Sim'],
        'Fumante': ['Não', 'Sim'],
        'Obeso': ['Não', 'Sim']
    }
)

# Ajustando a sensibilidade dos exames
cpd_dor = TabularCPD(
    variable='DorNoPeito',
    variable_card=2,
    values=[
        [0.5, 0.1],  # Dor = Não | Infarto = Não, Sim (mais específica)
        [0.5, 0.9]    # Dor = Sim | Infarto = Não, Sim (mais sensível)
    ],
    evidence=['Infarto'],
    evidence_card=[2],
    state_names={
        'DorNoPeito': ['Não', 'Sim'],
        'Infarto': ['Não', 'Sim']
    }
)

cpd_ecg = TabularCPD(
    variable='ECG_Anormal',
    variable_card=2,
    values=[
        [0.95, 0.05],  # ECG = Normal | Infarto = Não, Sim (mais específico)
        [0.05, 0.95]    # ECG = Anormal | Infarto = Não, Sim (mais sensível)
    ],
    evidence=['Infarto'],
    evidence_card=[2],
    state_names={
        'ECG_Anormal': ['Normal', 'Anormal'],
        'Infarto': ['Não', 'Sim']
    }
)

# 3. Adicionar as probabilidades ao modelo
modelo.add_cpds(cpd_fumante, cpd_obeso, cpd_infarto, cpd_dor, cpd_ecg)

# 4. Verificar o modelo
if not modelo.check_model():
    raise ValueError("Modelo inválido!")

# 5. Criar o mecanismo de inferência
inferencia = VariableElimination(modelo)

# 6. Função para imprimir resultados em português
def mostrar_diagnostico(variavel, evidencias=None):
    """
    Mostra os resultados de forma amigável em português
    """
    if evidencias:
        resultado = inferencia.query(variables=[variavel], evidence=evidencias)
        print(f"\n--- Diagnóstico para {variavel} dado: ---")
        for ev, val in evidencias.items():
            print(f"* {ev} = {val}")
    else:
        resultado = inferencia.query(variables=[variavel])
        print(f"\n--- Probabilidade geral de {variavel} ---")
    
    # Formatar a saída
    for estado, prob in zip(resultado.state_names[variavel], resultado.values):
        print(f"Probabilidade de {variavel} = {estado}: {prob*100:.2f}%")
    
    return resultado

# 7. Exemplos de diagnóstico
if __name__ == "__main__":
    print("\nSISTEMA DE DIAGNÓSTICO DE INFARTO")
    print("---------------------------------")
    
    # Caso 1: Probabilidade geral
    mostrar_diagnostico('Infarto')
    
    # Caso 2: Paciente com dor no peito
    mostrar_diagnostico('Infarto', {'DorNoPeito': 'Sim'})
    
    # Caso 3: Paciente obeso com ECG anormal
    mostrar_diagnostico('Infarto', {'Obeso': 'Sim', 'ECG_Anormal': 'Anormal'})
    
    # Caso 4: Paciente fumante com dor e ECG anormal
    mostrar_diagnostico('Infarto', {
        'Fumante': 'Sim',
        'DorNoPeito': 'Sim',
        'ECG_Anormal': 'Anormal'
    })
    
    # Caso 5: Qual a chance de ECG anormal em fumantes?
    mostrar_diagnostico('ECG_Anormal', {'Fumante': 'Sim'})
