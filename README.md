# Template de Projetos

Exemplo de estratégia para a disciplina PO-245.

## Organização do repositório

- [README.md](README.md): Apresenta o repositório
- [strategy](strategy/): estratégia escrita pelo aluno
- [main.ipynb](main.ipynb): Jupyter notebook com a documentação da sua estratégia, um exemplo de execução e a geração do relatório
- [data_market](data_market/): módulo provedor de dados
  - [datasets](datasets/): subdiretório com os dados disponibilizados para a estratégia
- [libs](libs/): bibliotecas externas adicionais
- [simulator](simulator/): código de apoio para simulação da estratégia
- [requirements.txt](requirements.txt): bibliotecas dependentes

## Primeiros passos

Recomendo primeiro criar um ambiente virtual (`venv`) utilizando `requirements.txt`. Este ambiente irá utilizar nossa versão corrigida do [QuantStat](#quantstat). Também recomendo utilizar vscode ou outra plataforma de desenvolvimento para facilitar o trabalho.

O principal arquivo é o Notebook `main.ipynb`, que deve conter:

- uma breve explicação da estratégia;
- a execução da estratégia;
- relatório para comparar a estratégia com um benchmark.

Note que a estratégia não é implementada no notebook. Apenas a invocamos no notebook para mantermos o código organizado.
A estratégia é contida na pasta `strategy` bem como todas as funções auxiliares para sua execução.

Rode o notebook por completo para entender a organização da solução.

Antes de rodar qualquer estratégia, preciamos carregar os dados. Prefira concentrar o tratamento de dados de maneira isolada da estratégia. Aqui armazenamos os dados e funções na pasta `data_market`.

Por fim, o notebook irá abrir uma janela perguntando onde salvar o relatório. Salve em uma pasta e abra em um browser. Você pode comparar sua estratégia com qualquer ticker compatível com o Yahoo Finance.

## QuantStat

Nosso template utiliza uma versão proprietári da biblioteca QuantStat disponível em https://github.com/fico-ita/quantstats/releases/tag/v0.0.63.

Este repo está configurado para utilizar a versão corrigida. Contudo, caso queira instalar manualmente, baixe o arquivo `.tar.gz` e instale via `pip`:

```bash
pip install quantstats-0.0.63.tar.gz
```
