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


class BadDataset(db.Model, QueryMixin):
    __tablename__ = 'bads_data_set'
    id = db.Column(db.Integer, primary_key=True)
    duration = db.Column(db.String(255))
    protocol_type = db.Column(db.String(255))
    service = db.Column(db.String(255))
    flag = db.Column(db.String(255))
    src_bytes = db.Column(db.String(255))
    dst_bytes = db.Column(db.String(255))
    land = db.Column(db.String(255))
    wrong_fragment = db.Column(db.String(255))
    urgent = db.Column(db.String(255))
    hot = db.Column(db.String(255))
    num_failed_logins = db.Column(db.String(255))
    logged_in = db.Column(db.String(255))
    num_compromised = db.Column(db.String(255))
    root_shell = db.Column(db.String(255))
    su_attempted = db.Column(db.String(255))
    num_root = db.Column(db.String(255))
    num_file_creations = db.Column(db.String(255))
    num_shells = db.Column(db.String(255))
    num_access_files = db.Column(db.String(255))
    num_outbound_cmds = db.Column(db.String(255))
    is_hot_login = db.Column(db.String(255))
    is_guest_login = db.Column(db.String(255))
    count = db.Column(db.String(255))
    srv_count = db.Column(db.String(255))
    serror_rate = db.Column(db.String(255))
    srv_serror_rate = db.Column(db.String(255))
    rerror_rate = db.Column(db.String(255))
    srv_rerror_rate = db.Column(db.String(255))
    same_srv_rate = db.Column(db.String(255))
    diff_srv_rate = db.Column(db.String(255))
    srv_diff_host_rate = db.Column(db.String(255))
    dst_host_count = db.Column(db.String(255))
    dst_host_srv_count = db.Column(db.String(255))
    dst_host_same_srv_rate = db.Column(db.String(255))
    dst_host_diff_srv_rate = db.Column(db.String(255))
    dst_host_same_src_port_rate = db.Column(db.String(255))
    dst_host_srv_diff_host_rate = db.Column(db.String(255))
    dst_host_serror_rate = db.Column(db.String(255))
    dst_host_srv_serror_rate = db.Column(db.String(255))
    dst_host_rerror_rate = db.Column(db.String(255))
    dst_host_srv_rerror_rate = db.Column(db.String(255))
    bad_class = db.Column(db.String(255))

    def __repr__(self):  # 相当于toString
        return '<TrainSet %r>' % self.id