from django.urls import path
from .views import run_script

urlpatterns = [
    path('run/', run_script, name='run_script'),
]

