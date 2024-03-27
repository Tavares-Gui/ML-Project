import functools

from . import create_app
from flask import (Blueprint, flash, g, redirect, render_template, request, session, url_for)

app = create_app()

bp = Blueprint('auth', __name__, url_prefix='/')

@bp.route('/', methods=('GET', 'POST'))
def register():
    return render_template('index.html')
