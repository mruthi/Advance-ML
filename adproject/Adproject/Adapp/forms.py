from django import forms
from .models import *


class AdForm(forms.ModelForm):
    class Meta():
        model=AdModel
        fields=['Age','EstimatedSalary','Gender_Male']
