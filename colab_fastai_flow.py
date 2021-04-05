import sys
 
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.vision.core import *
from ipywidgets import HBox,VBox,widgets,Button,Checkbox,Dropdown,Layout,Box,Output,Label,FileUpload
 
from google.colab     import drive
from google.colab     import widgets as colabwidigets
from IPython.display  import display,clear_output
from fastai.vision    import *
from fastai.vision.all import *
import ipywidgets as ipywidgets
 
from fastai.vision import widgets as fastwidgets
 
def drive_connect():
  drive.mount('/content/gdrive', force_remount=True)
  drive_connect.root_dir = "/content/gdrive/My Drive/"
  print('Google Drive conectado com sucesso')
 
 
def tab_data():
 
  drive_connect()
 
  # 1 - Definicao do Path
  display('Dataset/OK') ## temporario
  input_path = ipywidgets.Text(placeholder='Insira o path', value='')
  btn_path = ipywidgets.Button(description='Carregar Path')
  output_path = ipywidgets.Output()
  tab_data.path = ''
  box_path_input = ipywidgets.HBox([input_path, btn_path])
  box_path = ipywidgets.VBox([box_path_input, output_path])
 
  def on_btn_path_clicked(b):
    with output_path:
      tab_data.path = drive_connect.root_dir + input_path.value
      print('Path:\'' + tab_data.path + '\' definido com sucesso!')
      
  btn_path.on_click(on_btn_path_clicked)  
    
  # 2 - Ver imagem separada
  display('Dataset/OK/Jeriva/jeriva34.jpg') ## temporario
 
  input_path_img = ipywidgets.Text(placeholder='Insira o path da imagem', value='')
  btn_path_img = ipywidgets.Button(description='Ver imagem')
  output_path_img = ipywidgets.Output()
 
  box_path_input_img = ipywidgets.HBox([input_path_img, btn_path_img])
  box_path_img = ipywidgets.VBox([box_path_input_img, output_path_img])        
 
  def on_btn_path_img_clicked(b):
 
    with output_path_img:
      clear_output()
      tab_data.path_img = drive_connect.root_dir + input_path_img.value
      try:
        im = Image.open(tab_data.path_img).to_thumb(256,512)
        x = VBox([ipywidgets.HTML('Veja: ' + tab_data.path_img),
                  fastwidgets.widget(im, max_width="192px")])    
        display(x)
      except:
        print("Imagem não encontrada!")
      
  btn_path_img.on_click(on_btn_path_img_clicked)  
    
  # 4 - Splitter
  splitter_percent_validation = ipywidgets.FloatSlider(min = 0, max = 1, step = 0.01, value = 0.2, description = "% Validação")
 
  # 7 - Criar Dataloader
  tab_data.dls = None
  def build_data(path, splitter_percent_validation):
    folder = ['Aroeira-vermelha', 'Embauba', 'Jeriva', 'Olandi', 'Pitangueira'] # temporario
    data = get_image_files(path, folders=folder)
    splitter = RandomSplitter(valid_pct=splitter_percent_validation, seed=40)
    splitter(data)
    item_tfms = [Resize(456, method='crop')]
    batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]  
    blocks=(ImageBlock, CategoryBlock)
    get_image_files(path)    
    trees = DataBlock(blocks=blocks, 
                 get_items=get_image_files, 
                 splitter=splitter,
                 get_y=parent_label,
                 item_tfms=item_tfms,
                 batch_tfms=batch_tfms)
    dls = trees.dataloaders(path)
    tab_data.dls = dls
 
  btn_build_data = ipywidgets.Button(description='Criar Dataloader')
  output_build_data = ipywidgets.Output()
 
  def on_btn_build_data_clicked(b):
    with output_build_data:
      clear_output()
      build_data(tab_data.path, splitter_percent_validation.value)
      print('Dataloader criado com sucesso!')
      print('Parametros informados:')
      print('Path = ' + tab_data.path)
      print('% Valid. Spl. = ' + str(splitter_percent_validation.value))
 
  btn_build_data.on_click(on_btn_build_data_clicked)  
 
  box_build_data = ipywidgets.VBox([btn_build_data, output_build_data])        
 
  # 8 - Visualizar Batch
  splitter_show_batch = ipywidgets.IntSlider(min = 1, max = 10, step = 1, value = 3, description = "Itens do batch")
  btn_show_batch = ipywidgets.Button(description='Ver batch')
  output_show_batch = ipywidgets.Output()
  box_input_show_batch = ipywidgets.HBox([splitter_show_batch, btn_show_batch])
  box_show_batch = ipywidgets.VBox([box_input_show_batch, output_show_batch])        
 
  def on_btn_show_batch_clicked(b):
    y = tab_data.dls
 
    with output_show_batch:
      print('1')
      print(y)
      print('2')
      z = y.show_batch()
      print(2)
      
  btn_show_batch.on_click(on_btn_show_batch_clicked)  
  
  box_d_1 = box_path
  box_d_2 = box_path_img
  box_d_3 = ipywidgets.Box()
  box_d_4 = splitter_percent_validation
  box_d_5 = ipywidgets.Box()
  box_d_6 = ipywidgets.Box()
  box_d_7 = box_build_data
  box_d_8 = box_show_batch
  box_d_9 = ipywidgets.Box()
  box_d_10 = ipywidgets.Box()
  box_d_11 = ipywidgets.Box()
  
  
  ## Monta a tela
  data_children = [box_d_1,
                   box_d_2,
                   box_d_3,
                   box_d_4,
                   box_d_5,
                   box_d_6,
                   box_d_7,
                   box_d_8,
                   box_d_9,
                   box_d_10,
                   box_d_11]
 
  accordion_data = ipywidgets.Accordion(children=data_children)
 
  accordion_data.set_title(0, '1 - OK - Definir Path')
  accordion_data.set_title(1, '2 - DEV - Tratamento de imagens')
  accordion_data.set_title(2, '3 - X - Tratamento corretivo de imagen')
  accordion_data.set_title(3, '4 - DEV - Separação de conjunto de dados de treinamento e validação')
  accordion_data.set_title(4, '5 - X - Tratamento do dataset: augm_transform')
  accordion_data.set_title(5, '6 - X - Especificar batch size')
  accordion_data.set_title(6, '7 - DEV - Criar dataloader')
  accordion_data.set_title(7, '8 - DEV - Visualizar batch')
  accordion_data.set_title(8, '9 - X - Visarlizar infos do dataset')
  accordion_data.set_title(9, '10 - X - Visualizar as imagens do dataset / datacleaning')
  accordion_data.set_title(10, '11 - X - Novas definiccoes')
  
 
  display(accordion_data)
 
def tab_transfer_learning():
 
  def treinamento(model_architecture_value, epochs):
    dls = tab_data.dls
    defaults.callbacks[1] = Recorder(train_metrics=True)
    defaults.callbacks
    cbs=[ShowGraphCallback,ActivationStats(with_hist=True),SaveModelCallback]
    learn = None 
    print(model_architecture_value)
    learn = cnn_learner(dls,model_architecture_value,metrics=(error_rate, accuracy),cbs=cbs)
    #learn.show_training_loop()
    learn.fit_one_cycle(epochs,1e-2)
    dls.vocab
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(4, nrows=2, figsize = (25,5))  
 
  train_model = ipywidgets.Button(description="Criar Modelo e Treinar Rede")
 
  model_architecture = ipywidgets.Dropdown(options=[('alexnet', models.alexnet),
                                                      ('densenet121', models.densenet121),
                                                      ('densenet161', models.densenet161),
                                                      ('densenet169', models.densenet169),
                                                      ('densenet201', models.densenet201),
                                                      ('resnet18', models.resnet18),
                                                      ('resnet34', models.resnet34),
                                                      ('resnet50', models.resnet50),
                                                      ('resnet101', models.resnet101),
                                                      ('resnet152', models.resnet152),
                                                      ('squeezenet1_0', models.squeezenet1_0),
                                                      ('squeezenet1_1', models.squeezenet1_1),
                                                      ('vgg16_bn', models.vgg16_bn),
                                                      ('vgg19_bn',models.vgg19_bn)],
                                             description ='Arquitetura:',
                                             value = models.vgg19_bn)
 
  epochs = ipywidgets.IntSlider(min = 1,
                                  max = 50,
                                  step = 1,
                                  value = 3,
                                  continuous_update = False,
                                  description = "Épocas")
 
  #display(model_architecture,epochs,train_model)
 
  def on_button_clicked_train_model(b):
    treinamento(model_architecture.value, epochs.value)
 
  train_model.on_click(on_button_clicked_train_model)
 
  box_tl_12 = model_architecture
  box_tl_13 = epochs
  box_tl_14 = ipywidgets.Box()
  box_tl_15 = train_model
  
  ## Monta a tela
  data_children = [box_tl_12,
                   box_tl_13,
                   box_tl_14,
                   box_tl_15]
 
  accordion_tl = ipywidgets.Accordion(children=data_children)
 
  accordion_tl.set_title(0, '12 - DEV - Arquitetura')
  accordion_tl.set_title(1, '13 - DEV - Epochs')
  accordion_tl.set_title(2, '14 - X - Learning rate e momentum')
  accordion_tl.set_title(3, '15 - DEV - Executar Treinamento')  
 
  display(accordion_tl)
 
def tab_evaluetion_transfer_learning():
 
  box_etl_16 = ipywidgets.Box()
  box_etl_17 = ipywidgets.Box()
  box_etl_18 = ipywidgets.Box()
  box_etl_19 = ipywidgets.Box()
  box_etl_20 = ipywidgets.Box()
  box_etl_21 = ipywidgets.Box()
  box_etl_22 = ipywidgets.Box()
  box_etl_23 = ipywidgets.Box()
  
  ## Monta a tela
  data_children = [box_etl_16,
                   box_etl_17,
                   box_etl_18,
                   box_etl_19,
                   box_etl_20,
                   box_etl_21,
                   box_etl_22,
                   box_etl_23]
 
  accordion_etl = ipywidgets.Accordion(children=data_children)
 
  accordion_etl.set_title(0, '16 - X - Matriz de confusao')
  accordion_etl.set_title(1, '17 - X - Acuracia por categoria')
  accordion_etl.set_title(2, '18 - X - top losses')
  accordion_etl.set_title(3, '19 - X - most confused')  
  accordion_etl.set_title(4, '20 - X - Heatmaps')  
  accordion_etl.set_title(5, '21 - X - Visualizações alternativa')  
  accordion_etl.set_title(6, '22 - X - Data cleaning (de novo?)')  
  accordion_etl.set_title(7, '23 - X - Imagens no validation set')  
 
  display(accordion_etl)  
 
def exec():
 
  print('aqui')
 
  _tabs = ['DEV - Dados',
           'DEV - Transfer Learning',
           'X - Avaliação Transf. Learn.'#,
           #'Fine Tuning',
           #'Avaliação Fine Tun.',
           #'Predição',
           #'Deployment'
           ]
 
  _tabbar = colabwidigets.TabBar(_tabs)

   
  with _tabbar.output_to(0, select=('Dados')):
    tab_data()
 
  with _tabbar.output_to(1):
    tab_transfer_learning()    
 
  with _tabbar.output_to(2):
    tab_evaluetion_transfer_learning()    
 
  print('em baixo log')