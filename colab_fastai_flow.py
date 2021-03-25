!pip install fastai --upgrade -q

from fastai.vision.all  import *
from fastai.callback    import *
from fastai.metrics     import error_rate

from google.colab import widgets
from google.colab import output
from google.colab import drive

from ipywidgets import widgets as ipywidgets

def drive_connect():
  drive.mount('/content/gdrive', force_remount=True)
  drive_connect.root_dir = "/content/drive/My Drive/"
  print('Google Drive conectado com sucesso')

def get_path():
  str_path = str(set_path.output_variable.value)
  get_path.path_choice = str_path
  print('Path carregado')

def set_path():
  btn_set_path = ipywidgets.Button(description="Carregar Path")
  set_path.output_variable = ipywidgets.Text()
  display(set_path.output_variable, btn_set_path)

  def on_button_clicked_set_path(b):
      get_path()

  btn_set_path.on_click(on_button_clicked_set_path)
   
def dados(splitter_percent_validation, item_tfms_resize, item_tfms_resize_mtd, batch_tfms_aug_tfms_size):
  path = get_path.path_choice
  data = get_image_files(path)
  print(splitter_percent_validation, item_tfms_resize, item_tfms_resize_mtd, batch_tfms_aug_tfms_size)
  splitter = RandomSplitter(valid_pct = splitter_percent_validation, seed = 40)
  splitter(data)
  item_tfms = [Resize(item_tfms_resize, method=item_tfms_resize_mtd)]
  batch_tfms = [*aug_transforms(size=batch_tfms_aug_tfms_size), Normalize.from_stats(*imagenet_stats)]
  blocks=(ImageBlock, CategoryBlock)
  get_image_files(path)
  trees = DataBlock(blocks=blocks, 
                 get_items=get_image_files, 
                 splitter=splitter,
                 get_y=parent_label,
                 item_tfms=item_tfms,
                 batch_tfms=batch_tfms)
  dls = trees.dataloaders(path)
  dados.dls = dls
  print('dados carregados')
  

def treinamento(model_architecture_value, epochs):
  dls = dados.dls
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

def ui():

  t = widgets.TabBar(['G-Drive', 'Dados', 'Treinamento'])
  
  with t.output_to(0, select=False):
    drive_connect()

  with t.output_to(1, select=False):
    print('gdrive/My Drive/Dataset/OK')
    set_path()
    
    layout = ipywidgets.Layout(width='auto', height='auto')
    style = {'handle_color': 'blue'}

    btn_build_data = ipywidgets.Button(description="Criar datablock e dataloader")
    
    splitter_percent_validation = ipywidgets.FloatSlider(min = 0,
                                                         max = 1,
                                                         step = 0.01,
                                                         value = 0.2,
                                                         continuous_update = False,
                                                         layout = layout,
                                                         style = style,
                                                         description = "% Validação")

    item_tfms_resize = ipywidgets.IntSlider(min = 8,
                                            max = 640,
                                            step = 8,
                                            value = 456,
                                            continuous_update = False,
                                            layout = layout,
                                            style = style,
                                            description = "ITEM_TFMS RESIZE:")

    item_tfms_resize_mtd = ipywidgets.Dropdown(options=['crop', 'pad', 'squish'],
                                               value = 'crop',
                                               description = 'Metodo item_tfms_resize')
    
    batch_tfms_aug_tfms_size = ipywidgets.IntSlider(min = 128,
                                                    max = 480,
                                                    step = 8,
                                                    value = 224,
                                                    continuous_update = False,
                                                    layout = layout,
                                                    style = style,
                                                    description = "Batch_TFMS AUG_TFM_SIZE:")

    display(splitter_percent_validation,
            item_tfms_resize,
            item_tfms_resize_mtd,
            batch_tfms_aug_tfms_size,
            btn_build_data)

    def build_data(b):
      dados(splitter_percent_validation.value,
            item_tfms_resize.value,
            item_tfms_resize_mtd.value,
            batch_tfms_aug_tfms_size.value)

    btn_build_data.on_click(build_data)

    batch_show = ipywidgets.Button(description="Mostrar batch")

    batch_show_size = ipywidgets.IntSlider(min = 1,
                                           max = 50,
                                           step = 1,
                                           value = 5,
                                           continuous_update = False,
                                           layout = layout,
                                           style = style,
                                           description = "Qtd visualização batch: ")

    display(batch_show_size, batch_show)

    def on_button_clicked_batch_show(b):
      dados.dls.show_batch(max_n=batch_show_size.value)

    batch_show.on_click(on_button_clicked_batch_show)
    

  with t.output_to(2, select=False):
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
                                  layout = layout,
                                  style = style,
                                  description = "Épocas")

    display(model_architecture,
            epochs,
            train_model)

    def on_button_clicked_train_model(b):
      treinamento(model_architecture.value, epochs.value)

    train_model.on_click(on_button_clicked_train_model)