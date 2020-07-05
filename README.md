# 作业一的脚手架

## 这啥？

这是作业一的脚手架，包括：

- 一堆要测试的样本数据；
- 用于读入示例并写出结果的实用程序；
- 完整的项目结构CMakeLists.txt文件写你的CUDA程序。

更多细节内容可以在源码文件中看到。

## 咋用？

你通常只用修改`sources/src/core.h` and `sources/src/core.cu`。目前，这两个源代码文件包含一个演示示例，但该示例与hw1的要求无关。

当你准备提交你的作业1解决方案时，确保

- README文件被替换成你的实验报告，最好包括一个PDF版本
- 示例数据的结果被放入`results`文件夹
- 删掉示例数据文件`data.bin`和PPT`HW1.pdf`
- 文件夹的名字被命名为"廖浩淳-17341096"

最后，压缩文件夹为`.zip` 文件或者 `.7z` 文件，发送到multicoresysu2020@163.com截止于2020.07.05 23:59.

## 补充说明

1. 关于 hw1 的边界情况给同学们做一个补充说明：

   假如 x 元素周围只有 9 个元素（含 x 本身），这个时候概率按 n / 9 进行计算，即只考虑有效的元素。

2. 另外同学们的基础版本以及（多个）改进版本均需要提交，推荐的组织方式为给每一个版本提供一个文件夹，采用 sources_{tag} 的命名方式

   特别地，sources 应当包含你认为效果最好的版本

   偏好简洁的同学也可以通过提供多个 CMakeLists.txt 来区分版本，并做好标注

3. 如果你的项目编译方式和在课程主页上提供的案例有一定差异，请务必在 README 中写明

4. 有关代码结果的正确性，建议同学们同时实现一下串行版本进行校验。后续我会提供部分小规模 sample 的结果供同学们参考

5. 标注上每个分支是一个方案，串行版本跑不完

6. 

   

# Scaffold for hw1

## What is it?

This is a scaffold for your hw1, including:

- a bunch of sample data to test against;
- the utilities to read-in the samples and write-out your results;
- a full project structure with CMakeLists.txt for writing your CUDA program.

More details can be found in the source code files.

## How to use it?

You typically just need to modify `sources/src/core.h` and `sources/src/core.cu`. Currently, these two source code files contain a demonstrating example but the example has nothing to do with the requirements of hw1.

When you are about to hand-in your solutions to hw1, make sure
- this README file has been replaced by your experiment report (the report doesn't need to be a Markdown file and besides, you are always suggested to include a PDF version of your report);
- your results against the sample data have been put in the `results` folder;
- the sample data file `data.bin` has been removed;
- the folder name has been changed to `{your name}-{your ID}` (curly brackets are not needed).

And finally, compress the whole folder as a `.zip` file or a `.7z` file, and send it to multicoresysu2020@163.com before 2020.07.05 23:59.