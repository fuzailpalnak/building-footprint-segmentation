from building_segmentation.extractor import init_extractor
from building_segmentation.helpers.callbacks import CallbackList, get_default_callbacks
from building_segmentation.learner import Learner

extractor = init_extractor("binary")

model = extractor.load_model(model_name="AlBuNet")
criterion = extractor.load_criterion(criterion_name="BinaryCrossEntropy")
loader = extractor.load_loader(
    r"D:\Cypherics\Library\building-footprint-segmentation\data",
    "divide_by_255",
    "binary_label",
    2,
)
metrics = extractor.load_metrics(
    data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
)

optimizer = extractor.load_optimizer(model, "Adam")
callbacks = CallbackList(
    callbacks=get_default_callbacks(log_dir=r"D:\Cypherics\lib_check\out_data")
)
learner = Learner(
    model=model,
    criterion=criterion,
    loader=loader,
    metrics=metrics,
    callbacks=callbacks,
    optimizer=optimizer,
    scheduler=None,
)
learner.train(start_epoch=0, end_epoch=3)
