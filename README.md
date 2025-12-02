使用微信好友之间的聊天记录对模型进行训练，见证从0开始训练一个语言模型的过程，最后诞生一个模仿对方说话语气的ai ：）

（接下来这句话有点无聊，只是想表达一下我学了什么知识：这个项目使用了seq2seq + attention的语言模型，最早被Google用来做机器翻译，如果说机器翻译是将语言A翻译转换为语言B，这个模型就是将小A说的内容“翻译转换”为小B将会回应的内容。这是这个模型的本质，因此这个模型不是大语言模型，也不具备上下文记忆）

通过两个步骤，你可以得到一个模仿对方说话语气的ai。虽然步骤会有点麻烦，但是step1还有一个好处，就是为你提供了解码并导出微信聊天记录的方法！！！这个很好！！！如果你想做一些聊天内容分析的话将大有用处！！！

**----- step1：用于训练的聊天数据的准备：（苹果设备使用者可直接跳转第5步） -----**
1. 从微信将聊天记录迁移到苹果设备（iphone/ipad）：设置-聊天-聊天记录管理-导入与导出-导出到另一台手机或平板-后续操作按步骤完成即可
2. 下载apple deveice，并连接苹果设备，选择“将ipad上所有的数据备份到此电脑”，点击“立即备份”，然后点击下方右侧的“停止”，因为需要将默认备份地址进行修改

   <img width="1280" height="636" alt="image" src="https://github.com/user-attachments/assets/e9213d02-03c8-4409-960f-74eff0837112" />
   
3. 如果备份空间不足，按照这篇笔记将备份默认地址从C盘迁移到足够空间的位置（苹果设备使用者）：[iTunes怎么改默认备份位置 - 小红书](https://www.xiaohongshu.com/explore/667133030000000006005fe9?xsec_token=ABWbK-cKq3EU927x6aLRuvbde5DdbMNUENYwXEQznqqHg=&xsec_source=pc_search&source=unknown)
4. 备份默认地址迁移完成后，重新进行第2步，直到下方进度条完成
5. 从github下载微信聊天记录导出器wechat exporter，并按照说明文档操作即可导出（直接翻到下面看readme）：https://github.com/BlueMatthew/WechatExporter?tab=readme-ov-file
6. 打开wechat exporter，按照下方设置，仅导出聊天记录的文本
   
  <img width="200" height="100" alt="image" src="https://github.com/user-attachments/assets/f5e205b3-452f-40fe-a290-cffc0000b339" />
  <img width="200" height="130" alt="image" src="https://github.com/user-attachments/assets/603c8b96-bd73-4e15-bc33-aa114029a0b9" />
  
8. 将导出的聊天记录文件（.dat格式）使用word文档打开，并选择utf-8格式，可查看聊天记录

**----- step2：使用聊天数据进行训练，并进行有趣的模拟对话 -----**
1. 将导出的聊天记录文件（.dat格式）复制到与couple_chatbot_main文件相同的路径下
   
   <img width="200" height="250" alt="image" src="https://github.com/user-attachments/assets/25888e4f-63ba-409c-99e5-a758ad5ce93c" />
   
3. 使用编译器（如pycharm, vscode等）打开couple_chatbot_main文件，并运行
4. 输入：1. 当前目录下聊天记录（.dat文件）的文件名；2. 你想要扮演的一方的微信昵称；3. 请输入对方的微信昵称（对方将在训练后对你输入的内容进行回应）
   
   <img width="627" height="99" alt="image" src="https://github.com/user-attachments/assets/c26f2dcf-e52d-4214-9e36-2747af31d95f" />
   
6. 等待训练结束，训练过程中最好的模型会被立即保存
   
   <img width="1258" height="756" alt="image" src="https://github.com/user-attachments/assets/f07726f1-f99b-41ee-a0ca-9c1e30f6134b" />
   
8. 当有模型被保存以后，你可以运行couple_chatbot_inference开始聊天了！随着训练进行，模型的回应也会越来越真实！

HAVE FUN!!!
