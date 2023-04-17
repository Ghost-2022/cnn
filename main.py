#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/15 13:40
    @Auth: Jacob
    @Desc:
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    print(app.url_map)
    app.run(host='0.0.0.0')
