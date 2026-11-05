此目录不包含原始 SDF。请自行创建 raw/、free_base/ 或 metal/ 并提供有权使用的数据，
也可以通过脚本的 --input-dir 直接读取仓库外的数据目录，无需复制数据。

每个目录可以包含单分子或多分子 .sdf 文件；脚本只读取该目录顶层的 *.sdf。
Mol_classification.py 按文件分类：多分子文件中只要有一个分子含金属，整个文件归入 metal。
原始数据和运行结果已通过 .gitignore 排除。
