import numpy as np
import tensorflow as tf
import sys, traceback
import math
import os
from tensorflow.contrib.slim import fully_connected as fc
import time
from tensorflow.python.tools import inspect_checkpoint as chkp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as P
from mpl_toolkits.mplot3d import axes3d, Axes3D
from solvers.FullNonLinearBaseGPU import PDEFNLSolveBaseGPU


# use Z to calculate Gamma
class PDEFNLSolve2OptZGPU(PDEFNLSolveBaseGPU):

    def sizeNRG(self, iGam):
        if (iGam == self.nbStepGam):
            return 0
        else:
            return 1

    # calculate Gamma by conditionnel  expectation
    def buildGamStep(self, iStep, ListWeightUZ, ListBiasUZ, ListWeightGam, ListBiasGam, sess):
        is_trained_from_zero = True

        dic = {}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["XPrev"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='XPrev')
        dic["RandG"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d, self.sizeNRG(iStep)],
                                                name='randG')
        sample_size = tf.shape(dic["XPrev"])[0]
        sig = tf.constant(self.model.sigScal, dtype=tf.float32)
        sqrtsig = tf.constant(np.sqrt(self.model.sigScal), dtype=tf.float32)
        mu = self.model.muScal
        rescale = sig * np.sqrt(self.TStepGam * iStep)
        normX0 = tf.einsum('ij,j->ij', dic["XPrev"] - self.xInit - mu * self.TStepGam * iStep,
                           tf.math.reciprocal(rescale))
        if (iStep < self.nbStepGam):
            dic["Gam"] = self.networkGam.createNetworkWithInitializer(normX0, iStep, ListWeightGam[-1], ListBiasGam[-1],
                                                                      rescale)
            is_trained_from_zero = True
        else:
            dic["Gam"] = self.networkGam.createNetwork(normX0, iStep, rescale)
        GamTraj = tf.zeros([sample_size, self.d, self.d])
        sqrtDt = np.sqrt(self.TStepGam)
        if (self.sizeNRG(iStep) == 1):
            sqrtDt = np.sqrt(self.TStepGam)
            XNext = dic["XPrev"] + mu * self.TStepGam + tf.einsum('j,ij->ij', sig * sqrtDt, dic["RandG"][:, :, 0])
            XNextAnti = dic["XPrev"] + mu * self.TStepGam - tf.einsum('j,ij->ij', sig * sqrtDt, dic["RandG"][:, :, 0])
            if (iStep == self.nbStepGam - 1):
                GamTraj = 0.5 * (self.model.D2gTf(XNext) + self.model.D2gTf(XNextAnti))
            else:
                normX = tf.einsum('ij,j->ij', XNext - self.xInit - mu * self.TStepGam * (iStep + 1),
                                  tf.math.reciprocal(sig * np.sqrt(self.TStepGam * (iStep + 1))))
                _, Z = self.networkUZ.createNetworkNotTrainable(normX, iStep + 1, ListWeightUZ[-self.nbStepGamStab],
                                                                ListBiasUZ[-self.nbStepGamStab])
                normXAnti = tf.einsum('ij,j->ij', XNextAnti - self.xInit - mu * self.TStepGam * (iStep + 1),
                                      tf.math.reciprocal(sig * np.sqrt(self.TStepGam * (iStep + 1))))
                _, ZAnti = self.networkUZ.createNetworkNotTrainable(normXAnti, iStep + 2,
                                                                    ListWeightUZ[-self.nbStepGamStab],
                                                                    ListBiasUZ[-self.nbStepGamStab])

                GamTraj = 0.5 * tf.einsum('lij,j->lij', tf.einsum("li,lj->lij", Z - ZAnti, dic["RandG"][:, :, 0]),
                                          tf.math.reciprocal(sqrtDt * sig))
        else:
            GamTraj = self.model.D2gTf(dic["XPrev"])
        dic["weightLoc"], dic["biasLoc"] = self.networkGam.getBackWeightAndBias(iStep)
        dic["Loss"] = tf.reduce_mean(tf.pow(dic["Gam"] - GamTraj, 2))

        # Freeze and preload some weights
        if is_trained_from_zero:
            self.networkUZ.freeze_layers(iStep)
            self.networkGam.freeze_layers(iStep)
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate=dic["LRate"]).minimize(dic["Loss"])

        # Init weights
        sess.run(tf.compat.v1.global_variables_initializer())

        # Redo weights loading because variables initializing resets their values.
        if is_trained_from_zero:
            self.networkUZ.preload_weights(iStep, sess)
            self.networkGam.preload_weights(iStep, sess)

        return dic

    # calculate Gamma by conditionnal expectation
    def buildGamStep0(self, ListWeightUZ, ListBiasUZ, ListWeightGam, ListBiasGam, Gam0_initializer, sess):
        dic = {}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["RandG"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d, 1], name='randG')
        #from networks.global_vars import CKPT_PATH_GAMMA
        #from networks.tf_utils import load_tensor_from_ckpt
        print(f'input Gam0_initializer={Gam0_initializer}')
        #if CKPT_PATH_GAMMA is not None:
           # Gam0_initializer = load_tensor_from_ckpt(CKPT_PATH_GAMMA, 'Gam0')
            #print('Successfully loaded Gam0 weights.')
            #print(f'Gam0_initializer={Gam0_initializer}')
            #Gam0_initializer = tf.constant_initializer(Gam0_initializer)

        dic["Gam0"] = tf.compat.v1.get_variable("Gam0", [self.d, self.d], tf.float32, Gam0_initializer)
        sample_size = tf.shape(dic["RandG"])[0]
        sig = tf.constant(self.model.sigScal, dtype=tf.float32)
        sqrtsig = tf.constant(np.sqrt(self.model.sigScal), dtype=tf.float32)
        mu = self.model.muScal
        sqrtDt = np.sqrt(self.TStepGam)
        GamTraj = tf.zeros([sample_size, self.d, self.d])
        XNext = self.xInit + mu * self.TStepGam + tf.einsum('j,ij->ij', sig, sqrtDt * dic["RandG"][:, :, 0])
        XNextAnti = self.xInit + mu * self.TStepGam - tf.einsum('j,ij->ij', sig, sqrtDt * dic["RandG"][:, :, 0])
        normX = tf.einsum('ij,j->ij', XNext - self.xInit - mu * self.TStepGam,
                          tf.math.reciprocal(sig * np.sqrt(self.TStepGam)))
        _, Z = self.networkUZ.createNetworkNotTrainable(normX, 1, ListWeightUZ[-self.nbStepGamStab],
                                                        ListBiasUZ[-self.nbStepGamStab])
        normXAnti = tf.einsum('ij,j->ij', XNextAnti - self.xInit - mu * self.TStepGam,
                              tf.math.reciprocal(sig * np.sqrt(self.TStepGam)))
        _, ZAnti = self.networkUZ.createNetworkNotTrainable(normXAnti, 2, ListWeightUZ[-self.nbStepGamStab],
                                                            ListBiasUZ[-self.nbStepGamStab])
        GamTraj = 0.5 * tf.einsum('lij,j->lij', tf.einsum("li,lj->lij", Z - ZAnti, dic["RandG"][:, :, 0]),
                                  tf.math.reciprocal(sqrtDt * sig))
        dic["Loss"] = tf.reduce_mean(tf.pow(dic["Gam0"] - GamTraj, 2))

        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate=dic["LRate"]).minimize(dic["Loss"])
        # initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        # Load weights if needed
       # from networks.global_vars import CKPT_PATH_BSDE_UZSTEP0, CKPT_PATH_BSDE_UZSTEP0_STEPVAL
        #if CKPT_PATH_BSDE_UZSTEP0 is not None:
            #self.networkUZ.preload_weights(CKPT_PATH_BSDE_UZSTEP0_STEPVAL, sess, name_of_scope=f'NetWorkUZNonTrain_', weights_path=CKPT_PATH_BSDE_UZSTEP0)
        return dic
