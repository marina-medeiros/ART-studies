# ARTMAP
Enquanto modelos ART mais básicos organizam padrões em categorias de forma não supervisionada, o ARTMAP conecta duas redes ART de modo que seja possível mapear entradas para classes.

As entradas do algoritmo são um par $x^a$ e $x^b$.

- $ART_a$: recebe as entradas $x^a$.
- $ART_b$: recebe os rótulos $x^b.

## Treinamento:
- Cada rede tenta categorizar sua entrada
- A *função match* no *map field* verifica se a categoria escolhida em $ART_a$ está corretamente associada ao rótulo em $ART_b$.
- O teste de vigilância do *map field* verifica se a correspondência entre entrada e rótulo é consistente.
    - Se for, o aprendizado ocorre.
    - Se não for, a categoria de $ART_a$ é inibida e busca-se outra. Se nenhuma for suficiente, outra categoria será criada.
- Se uma categoria foi aceita, atualiza-se o vetor de pesos usando $\beta$ como um dos parâmetros.


