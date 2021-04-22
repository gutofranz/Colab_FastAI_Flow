#Importacao via python da biblioteca fastai
#! [ -e /content ] && pip install -Uqq fastai  

#Importações do google colab
from google.colab import drive
from google.colab import widgets as colabwidigets
from google.colab import output  as colaboutputs

#Importações Ipywidigets
from ipywidgets import widgets as ipywidgets

#Importações Fastai
from fastai.data.all import *
from fastai.vision.core import *
from fastai.vision    import *
from fastai.vision.all import *

# Função para a conexão com o GDrive
def drive_connect():
  drive.mount('/content/gdrive', force_remount=True)
  drive_connect.root_dir = "/content/gdrive/My Drive/"
  print('Google Drive conectado com sucesso')

def set_path():
  set_path.path = None

def set_dls():
  set_dls.dls = None 

def set_learner():
  set_learner.learner = None


# Função para a criação do DataLoader
def build_data(path, item_tfms_resize, item_tfms_resize_mtd, splitter_percent_validation, batch_tfms_aug_tfms_size):
  #folder = ['Aroeira-vermelha', 'Embauba', 'Jeriva', 'Olandi', 'Pitangueira'] # temporario
  data = get_image_files(path)#, folders=folder)

  # Splitter
  splitter = RandomSplitter(valid_pct=splitter_percent_validation, seed=40)
  splitter(data)

  # Transformações
  item_tfms = [Resize(item_tfms_resize, method=item_tfms_resize_mtd)]
  batch_tfms=[*aug_transforms(size=batch_tfms_aug_tfms_size), Normalize.from_stats(*imagenet_stats)]  

  # Criação do DataBlock
  blocks=(ImageBlock, CategoryBlock)
  get_image_files(path)    
  trees = DataBlock(blocks=blocks,
                    get_items=get_image_files,
                    splitter=splitter,
                    get_y=parent_label,
                    item_tfms=item_tfms,
                    batch_tfms=batch_tfms)
  
  # Criação do DataLoader
  dls = trees.dataloaders(path)
  build_data.dls = dls

# Função para a realização do treinamento
def treinamento(model_architecture, epochs):
  dls = set_dls.dls
  defaults.callbacks[1] = Recorder(train_metrics=True)
  defaults.callbacks
  cbs=[ShowGraphCallback,ActivationStats(with_hist=True),SaveModelCallback]
  learn = None 
  #print(model_architecture_value)
  learn = cnn_learner(dls,model_architecture,metrics=(error_rate, accuracy),cbs=cbs)
  #learn.show_training_loop()
  learn.fit_one_cycle(epochs,1e-2)
  treinamento.learner = learn

# Função para executar a Interface Visual
def exec():

  # Criação do painel que vai conter as abas
  tab = colabwidigets.TabBar(['Dados', 'Transfer Learning', 'Av. des. TL'])

  # Execução da primeira aba: Dados
  with tab.output_to(0):
    
    # Executa a conexão com o GDrive
    drive_connect()
    
    # Criação do grid que vai conter as etapas e entradas/saídas
    grid = colabwidigets.Grid(14, 1)  

    # Atividade 1 - Definir Path
    with grid.output_to(0,0):
      print ('1 - Definir Path')

      display('Dataset/OK') ## temporario

      input_path = ipywidgets.Text(placeholder='Insira o path', value='')
      btn_path = ipywidgets.Button(description='Carregar Path')

      display(input_path, btn_path)

      def on_btn_path_clicked(b):
        # Saída - Atividade 1
        with grid.output_to(1,0):
          grid.clear_cell()
          set_path.path = drive_connect.root_dir + input_path.value
          print('Path:\'' + set_path.path + '\' definido com sucesso!')
      
      btn_path.on_click(on_btn_path_clicked)  

    # Atividade 2 - Tratamento de imagens
    with grid.output_to(2,0):
      print ('2 - Tratamento de imagens')
      
      item_tfms_resize = ipywidgets.IntSlider(min = 8, max = 640, step = 8, value = 456, description = "ITEM_TFMS RESIZE:")
      item_tfms_resize_mtd = ipywidgets.Dropdown(options=['crop', 'pad', 'squish'],value = 'crop',description = 'Metodo item_tfms_resize')

      display(item_tfms_resize, item_tfms_resize_mtd)
    
    # Atividade 3 - Tratamento corretivo de imagen
    with grid.output_to(3,0):
      print ('3 - X - Tratamento corretivo de imagen')
      
    # Atividade 4 - Separação de conjunto de dados de treinamento e validação
    with grid.output_to(4,0):
      print ('4 - Separação de conjunto de dados de treinamento e validação')
      
      splitter_percent_validation = ipywidgets.FloatSlider(min = 0, max = 1, step = 0.01, value = 0.2, description = "% Validação")

      display(splitter_percent_validation)

    # Atividade 5 - Tratamento do dataset: augm_transform
    with grid.output_to(5,0):
      print ('5 - DEV - Tratamento do dataset: augm_transform')
      
      batch_tfms_aug_tfms_size = ipywidgets.IntSlider(min = 128, max = 480, step = 8, value = 224, description = "Batch_TFMS AUG_TFM_SIZE:")

      display(batch_tfms_aug_tfms_size)

    # Atividade 6 - Especificar batch size
    with grid.output_to(6,0):
      print ('6 - X - Especificar batch size')

    # Atividade 7 - Criar dataloader
    with grid.output_to(7,0):
      print ('7 - DEV - Criar dataloader')    

      btn_build_data = ipywidgets.Button(description='Criar Dataloader')

      display(btn_build_data)

      def on_btn_build_data_clicked(b):
        # Saída - Atividade 6
        with grid.output_to(8,0):
          build_data(set_path.path, item_tfms_resize.value, item_tfms_resize_mtd.value, splitter_percent_validation.value, batch_tfms_aug_tfms_size.value)
          set_dls.dls = build_data.dls
          print('Dataloader criado com sucesso!')
          print('Parametros informados:')
          print('Path = ' + set_path.path)
          print('item_tfms_resize = ' + str(item_tfms_resize.value))
          print('item_tfms_resize_mtd = ' + str(item_tfms_resize_mtd.value))
          print('batch_tfms_aug_tfms_size = ' + str(batch_tfms_aug_tfms_size.value))
          print('% Valid. Spl. = ' + str(splitter_percent_validation.value))
 
      btn_build_data.on_click(on_btn_build_data_clicked)  
    
    # Atividade 8 - Visualizar batch
    with grid.output_to(9,0):
      print ('8 - Visualizar batch')

      slider_show_batch = ipywidgets.IntSlider(min = 1, max = 10, step = 1, value = 3, description = "Itens do batch")
      btn_show_batch = ipywidgets.Button(description='Ver batch')

      display(slider_show_batch, btn_show_batch)
    
      def on_btn_show_batch_clicked(b):
        # Saída - Atividade 8
        with grid.output_to(10,0):
          grid.clear_cell()
          y = build_data.dls
          print('Carregando batch, aguarde...')
          y.show_batch(max_n=slider_show_batch.value)
        
      btn_show_batch.on_click(on_btn_show_batch_clicked)  

    # Atividade 9 - Visarlizar infos do dataset
    with grid.output_to(11, 0):
      print ('9 - X - Visarlizar infos do dataset')

    # Atividade 10 - Visualizar as imagens do dataset / datacleaning
    with grid.output_to(12, 0):
      print ('10 - X - Visualizar as imagens do dataset / datacleaning')

    # Atividade 11 - Novas definiccoes
    with grid.output_to(13, 0):
      print ('11 - X - Novas definiccoes')
  
  # Execução da segunda aba: Treinamento do modelo - #1 Transfer Learning
  with tab.output_to(1, select=False):  

    # Criação do grid que vai conter as etapas e entradas/saídas
    grid2 = colabwidigets.Grid(5,1)

    # Atividade 12 - Arquitetura
    with grid2.output_to(0,0):
      print ('12 - Arquitetura')

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
      
      display(model_architecture)

    # Atividade - 13 - Epochs
    with grid2.output_to(1,0):
      print ('13 - DEV - Epochs')

      slider_epochs = ipywidgets.IntSlider(min = 1, max = 50, step = 1, value = 3, description = "Epochs")
    
      display(slider_epochs)

    # Atividade - 14 - Learning rate e momentum
    with grid2.output_to(2,0):
      print ('14 - X - Learning rate e momentum')

    # Atividade - 15- Executar Treinamento
    with grid2.output_to(3,0):
      print ('15 - Executar Treinamento')  

      btn_train_model = ipywidgets.Button(description='Rodar treinamento')    

      display(btn_train_model)

      def on_btn_train_model_clicked(b):
        # Saída - Atividade 15
        with grid2.output_to(4,0):
          grid2.clear_cell()
          print('-- Rodando treinamento -- ')
          print('Arquitetura = ' + str(model_architecture.value))
          print('Epochs = ' + str(slider_epochs.value))
          treinamento(model_architecture.value, slider_epochs.value)   
          set_learner.learner = treinamento.learner     
 
      btn_train_model.on_click(on_btn_train_model_clicked)

  # Execução da terceira aba: Avaliação do desempenho do Transfer Learning
  with tab.output_to(2, select=False):  

    # Criação do grid que vai conter as etapas e entradas/saídas
    grid3 = colabwidigets.Grid(10,1)

    # Atividade - 16 - X - Matriz de confusao
    with grid3.output_to(0,0):
      print('16 - X - Matriz de confusao')

      btn_show_c_matrix = ipywidgets.Button(description='Ver Matriz de confusão')

      #display(btn_show_c_matrix)
    
      def on_btn_show_c_matrix(b):
        # Saída - Atividade 8
        with grid3.output_to(1,0):
          grid3.clear_cell()
          y = set_learner.learn
          print('Carregando Matriz de confusão, aguarde...')
          interp = ClassificationInterpretation.from_learner(y)
          interp.plot_confusion_matrix()
        
      btn_show_c_matrix.on_click(on_btn_show_c_matrix)  

    # Atividade - 17 - Acuracia por categoria
    with grid3.output_to(2,0):
      print('17 - X - Acuracia por categoria')

    # Atividade - 18 - top losses
    with grid3.output_to(3,0):
      print('18 - X - top losses')

    # Atividade - 19 - most confused
    with grid3.output_to(4,0):
      print('19 - X - most confused')  

    # Atividade - 20 - Heatmaps
    with grid3.output_to(5,0):
      print('20 - X - Heatmaps')  
    
    # Atividade - 21 - Visualizações alternativa  
    with grid3.output_to(6,0):
      print('21 - X - Visualizações alternativa')  
    
    # Atividade - 22 - Data cleaning (de novo?)
    with grid3.output_to(7,0):
      print('22 - X - Data cleaning (de novo?)')  
    
    # Atividade - 
    with grid3.output_to(8,0):
      print('23 - X - Imagens no validation set')  




#dls.vocab


#interp.plot_top_losses(4, nrows=2, figsize = (25,5)
print('foi')    