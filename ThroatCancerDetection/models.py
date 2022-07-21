from django.db import models

# Create your models here.
class FilesAdmin(models.Model):
    fileUpload = models.FileField(null=True)