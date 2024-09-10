from django.db import models

# Create your models here.
class AdModel(models.Model):

    Age=models.FloatField()
    EstimatedSalary=models.FloatField()
    Gender_Male=models.FloatField()
