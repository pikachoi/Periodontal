import re
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .models import Doctor, Doctor_profile, Question
from .forms import LoginForm, SignUpForm, IDSearchForm, PasswordSearchForm, PasswordResetForm
from django.views.decorators.cache import never_cache
from django.contrib import messages

@never_cache
def login_view(request):
    if request.user.is_authenticated:
        return render(request, "Accounts_login_status.html")
    
    if request.method == "POST":
        form = LoginForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                return redirect("diagnosis_single")
            else:
                form.add_error(None, "등록되지 않은 아이디이거나 아이디 또는 비밀번호를 잘못 입력했습니다.")
    else:
        form = LoginForm()

    context = {"form": form}
    return render(request, "Accounts_login.html", context)


def logout_view(request) :
    logout(request)
    return redirect("login")


@never_cache
def signup_view(request):
    if request.user.is_authenticated:
        return render(request, "Accounts_login_status.html")
        
    question_list = [q.question for q in Question.objects.all()]

    if request.method == 'POST':
        form = SignUpForm(request.POST)
        
        if form.is_valid():
            user = Doctor.object.create_user(
                username=form.cleaned_data['username'],
                password=form.cleaned_data['password1'],
            )

            Doctor_profile.objects.create(
                doctor_username=user,
                license=form.cleaned_data['License_Number'],
                real_name=form.cleaned_data['name'],
                email=form.cleaned_data['email'],
                phone=form.cleaned_data['phonenumber'],
                belong=form.cleaned_data['Affiliation'],
                position=form.cleaned_data['Rank'],
                medical_subject=form.cleaned_data['Subject'],
                question=Question.objects.get(question = request.POST["question"]),
                question_answer=form.cleaned_data['hint'],
            )
            messages.success(request, "회원가입이 완료되었습니다. 로그인 해주세요.")
            return redirect("login")
    else:
        form = SignUpForm()

    return render(request, "Accounts_signup.html", {"form": form, "question": question_list})


@never_cache
def id_search_view(request):
    if request.user.is_authenticated:
        return render(request, "Accounts_login_status.html")
    
    question_list = [q.question for q in Question.objects.all()]

    form = IDSearchForm(request.POST or None)
    
    if request.method == "POST" and form.is_valid():
        name = form.cleaned_data["name"]
        email = form.cleaned_data["email"]
        license_number = form.cleaned_data["License_Number"]
        question = form.cleaned_data["question"].question
        hint = form.cleaned_data["hint"]
        
        doctor_profiles = Doctor_profile.objects.filter(
            question__question=question, question_answer=hint,
            real_name=name, email=email, license=license_number)
        
        if doctor_profiles.exists():
            if doctor_profiles.count() == 1:
                profile = doctor_profiles.first()
                return render(request, "Accounts_id_search.html",
                              {"real_name": f'{profile.real_name} 님의 아이디는', "doctor_username": f'{profile.doctor_username.username} 입니다.'})
            else:
                # 일치하는 프로필이 여러 개인 경우(중복 프로필은 애초에 가입이 되지 않으나 버그, 보안 대비)
                return render(request, "Accounts_id_search.html",
                              {"matching_profiles": doctor_profiles})
        else:
            form.add_error(None, "일치하는 정보가 없습니다. 다시 입력해주세요.")
    
    return render(request, "Accounts_id_search.html", {"form": form, "question": question_list})


@never_cache
def password_search_view(request):
    if request.user.is_authenticated:
        return render(request, "Accounts_login_status.html")
    
    question_list = [q.question for q in Question.objects.all()]
    
    form = PasswordSearchForm(request.POST or None)
    
    if request.method == "POST" and form.is_valid():
        username = form.cleaned_data["username"]
        question = form.cleaned_data["question"]
        name = form.cleaned_data["name"]
        email = form.cleaned_data["email"]
        license_number = form.cleaned_data["License_Number"]
        hint = form.cleaned_data["hint"]
        
        doctor_profiles = Doctor_profile.objects.filter(
            doctor_username__username=username,
            question=question, question_answer=hint,
            real_name=name, email=email, license=license_number
        )
        
        if doctor_profiles.exists():
            if doctor_profiles.count() == 1:
                request.session["reset_username"] = doctor_profiles[0].doctor_username.username
                return redirect("password_reset")
            else:
                return render(request, "Accounts_password_search.html", {"form": form, "matching_profiles": doctor_profiles})
        else:
            form.add_error(None, "일치하는 정보가 없습니다. 다시 입력해주세요.")
    
    return render(request, "Accounts_password_search.html", {"form": form,  "question": question_list})


@never_cache
def password_reset_view(request):
    if request.user.is_authenticated:
        return render(request, "Accounts_login_status.html")

    # 세션에 정보가 없다면 페이지 접근 제한(url 수동 입력으로 접근을 막기위함)
    username = request.session.get("reset_username")
    if not username:
        return redirect("password_search")

    form = PasswordResetForm(request.POST or None)

    if form.is_valid():
        password1 = form.cleaned_data["password1"]
        
        try:
            doctor = Doctor.object.get(username=username)
            doctor.set_password(password1)
            doctor.save()
            request.session.pop("reset_username")  # 비밀번호 재설정 완료 후 세션에서 삭제
            messages.success(request, "비밀번호가 재설정되었습니다.")
            return redirect("login")
        
        except Doctor.DoesNotExist:
            form.add_error(None, "사용자를 찾을 수 없습니다.")

    return render(request, "Accounts_password_reset.html", {"form": form})