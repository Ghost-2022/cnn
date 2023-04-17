#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Date: 2023/4/15 16:15
    @Auth: Jacob
    @Desc:
"""

from app.common.db import db


class QueryMixin:
    def get_info(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns if hasattr(self, c.name)}



class TrainSetModel(db.Model, QueryMixin):
    __tablename__ = 'train_set'

    id = db.Column(db.Integer, primary_key=True)
    feature_1 = db.Column(db.String(255))
    feature_2 = db.Column(db.String(255))
    feature_3 = db.Column(db.String(255))
    feature_4 = db.Column(db.String(255))
    feature_5 = db.Column(db.String(255))
    feature_6 = db.Column(db.String(255))
    feature_7 = db.Column(db.String(255))
    feature_8 = db.Column(db.String(255))
    feature_9 = db.Column(db.String(255))
    feature_10 = db.Column(db.String(255))
    feature_11 = db.Column(db.String(255))
    feature_12 = db.Column(db.String(255))
    feature_13 = db.Column(db.String(255))
    feature_14 = db.Column(db.String(255))
    feature_15 = db.Column(db.String(255))
    feature_16 = db.Column(db.String(255))
    feature_17 = db.Column(db.String(255))
    feature_18 = db.Column(db.String(255))
    feature_19 = db.Column(db.String(255))
    feature_20 = db.Column(db.String(255))
    feature_21 = db.Column(db.String(255))
    feature_22 = db.Column(db.String(255))
    feature_23 = db.Column(db.String(255))
    feature_24 = db.Column(db.String(255))
    feature_25 = db.Column(db.String(255))
    feature_26 = db.Column(db.String(255))
    feature_27 = db.Column(db.String(255))
    feature_28 = db.Column(db.String(255))
    feature_29 = db.Column(db.String(255))
    feature_30 = db.Column(db.String(255))
    feature_31 = db.Column(db.String(255))
    feature_32 = db.Column(db.String(255))
    feature_33 = db.Column(db.String(255))
    feature_34 = db.Column(db.String(255))
    feature_35 = db.Column(db.String(255))
    feature_36 = db.Column(db.String(255))
    feature_37 = db.Column(db.String(255))
    feature_38 = db.Column(db.String(255))
    feature_39 = db.Column(db.String(255))
    feature_40 = db.Column(db.String(255))
    feature_41 = db.Column(db.String(255))
    feature_42 = db.Column(db.String(255))

    def __repr__(self):  # 相当于toString
        return '<TrainSet %r>' % self.id


class TestSetModel(db.Model, QueryMixin):
    __tablename__ = 'test_set'
    id = db.Column(db.Integer, primary_key=True)
    feature_1 = db.Column(db.String(255))
    feature_2 = db.Column(db.String(255))
    feature_3 = db.Column(db.String(255))
    feature_4 = db.Column(db.String(255))
    feature_5 = db.Column(db.String(255))
    feature_6 = db.Column(db.String(255))
    feature_7 = db.Column(db.String(255))
    feature_8 = db.Column(db.String(255))
    feature_9 = db.Column(db.String(255))
    feature_10 = db.Column(db.String(255))
    feature_11 = db.Column(db.String(255))
    feature_12 = db.Column(db.String(255))
    feature_13 = db.Column(db.String(255))
    feature_14 = db.Column(db.String(255))
    feature_15 = db.Column(db.String(255))
    feature_16 = db.Column(db.String(255))
    feature_17 = db.Column(db.String(255))
    feature_18 = db.Column(db.String(255))
    feature_19 = db.Column(db.String(255))
    feature_20 = db.Column(db.String(255))
    feature_21 = db.Column(db.String(255))
    feature_22 = db.Column(db.String(255))
    feature_23 = db.Column(db.String(255))
    feature_24 = db.Column(db.String(255))
    feature_25 = db.Column(db.String(255))
    feature_26 = db.Column(db.String(255))
    feature_27 = db.Column(db.String(255))
    feature_28 = db.Column(db.String(255))
    feature_29 = db.Column(db.String(255))
    feature_30 = db.Column(db.String(255))
    feature_31 = db.Column(db.String(255))
    feature_32 = db.Column(db.String(255))
    feature_33 = db.Column(db.String(255))
    feature_34 = db.Column(db.String(255))
    feature_35 = db.Column(db.String(255))
    feature_36 = db.Column(db.String(255))
    feature_37 = db.Column(db.String(255))
    feature_38 = db.Column(db.String(255))
    feature_39 = db.Column(db.String(255))
    feature_40 = db.Column(db.String(255))
    feature_41 = db.Column(db.String(255))
    feature_42 = db.Column(db.String(255))

    def __repr__(self):  # 相当于toString
        return '<TrainSet %r>' % self.id
