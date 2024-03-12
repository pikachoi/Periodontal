from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

class DoctorManager(BaseUserManager) :
    def create_user(self, username,  password = None) :
        if not username :
            raise ValueError("username")
        
        user = self.model(
            username = username,
        )
        user.set_password(password)
        user.save(using = self._db)
        return user

    def create_superuser(self, username, password = None) :
        user = self.create_user(
            username = username,
            password = password,
        )
        user.is_active = True
        user.is_staff = True
        user.is_superuser = True
        user.save(using = self._db)
        return user

class Doctor(AbstractBaseUser) :
    username        = models.CharField(max_length = 128, primary_key = True)
    last_login      = models.DateTimeField(verbose_name = "last login", auto_now = True)
    is_active       = models.BooleanField(default = True)
    is_staff        = models.BooleanField(default = False)
    is_superuser    = models.BooleanField(default = False)
    
    object = DoctorManager()

    USERNAME_FIELD = "username"
    REQUIRED_FIELDS = []
    
    def __str__(self):
        return self.username

    def has_perm(self, perm, obj=None):
        return self.is_staff
 
    def has_module_perms(self, app_lable):
        return True

class Question(models.Model) :
    idx             = models.AutoField(primary_key = True)
    question        = models.CharField(max_length = 256)

class Doctor_profile(models.Model) :
    doctor_username = models.OneToOneField(Doctor, on_delete = models.CASCADE, primary_key = True, db_column = "doctor_username")
    license         = models.CharField(max_length = 128, unique = True)
    real_name       = models.CharField(max_length = 32, default = "홍길동")
    email           = models.EmailField(max_length = 128)
    phone           = models.CharField(max_length = 13, default = "010-0000-0000")
    belong          = models.CharField(max_length = 128)    # 소속
    position        = models.CharField(max_length = 128)    # 직급/ 직위
    medical_subject = models.CharField(max_length = 128)    # 진료 과목
    question        = models.ForeignKey(Question, on_delete = models.CASCADE, default = 1, db_column = "question")
    question_answer = models.CharField(max_length = 128)