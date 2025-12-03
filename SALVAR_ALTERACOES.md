# Como salvar alterações

1. Verifique o estado do repositório para ver os ficheiros modificados:
   ```bash
   git status -sb
   ```
2. Adicione os ficheiros que quer guardar ao índice:
   ```bash
   git add <caminho/do/ficheiro>
   ```
   Para adicionar tudo o que mudou:
   ```bash
   git add .
   ```
3. Guarde as alterações num commit com uma mensagem descritiva:
   ```bash
   git commit -m "Mensagem curta sobre a alteração"
   ```
4. Se quiser enviar o commit para o repositório remoto, faça o push:
   ```bash
   git push origin <ramo>
   ```

> Dica: use mensagens de commit claras para facilitar o histórico (ex.: "Corrige pipeline de treino").
