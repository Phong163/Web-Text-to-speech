from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('',views.get_home),
    path('process_text', views.process_text, name='process_text'),
]