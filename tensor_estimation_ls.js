/*Estimating the Tensor using the LS ( Least Square method )*/
define([
    'dMRIJS/core',
    'when/when',
    'mythread',
    'dMRIJS/volume',
    'numeric'
], function ( dMRIJS, when, myThreadJS ) {
    var undefined;

    var Thread = myThreadJS.Thread,
        browserMethod = Thread.browserMethod,
        AbstractSpace = myThreadJS.AbstractSpace,
        Space = myThreadJS.Space,
        VolumeSpace = myThreadJS.VolumeSpace,
        Volume3 = dMRIJS.Volume3,
        Volume33 = dMRIJS.Volume33,
        //The numeric module URL
        numericURL = require.toUrl( 'numeric' ) + '.js';
    
    //The Sequential Version of the Tensor Estimation for LS method class
    var TensorEstimationSeqLS = myThreadJS.BaseObject.extend( 'dMRIJS.TensorEstimationSeqLS', {
      
        //Class constructor
        constructor: function ( data ) {
            //Retrieve the Volume class
            var volumes = data.volumes,
                volumesNumber = data.bval.length,
                mask = data.mask;

            /*Initialize the parameters of the tensor estimation*/
            this.bValues = data.bval;
            this.bVectors = data.bvec;
            this.volumesData = volumes;
            this.volumeMask = mask;
            this.threshold = data.threshold;

            //Call the Thread base constructor
            this.$super( 'myThreadJS.Thread', this )({
                input_data: data
            });

            /*Set Computation type to "local", if Transferrables is not supported.
              This helps to avoid lost computation time in data copy.*/ 
            if( !this.ALLOW_TRANSFERRABLE ) {   
                this.setType( 'local' );
            }
        },

        //Get the B matrix 
        B: function ( bVectors ) {
            var B = [],
                bVector,
                bRow,
                vectorNumber = bVectors.length;

            for( var i = 1; i < vectorNumber; i++ ) {
                //bVec i
                bVector = bVectors[ i ];
                
                //B Row [b11, b22, b33, b12, b13, b23]
                bRow = [ 
                    Math.pow( bVector[ 0 ], 2 ), Math.pow( bVector[ 1 ], 2 ), Math.pow( bVector[ 2 ], 2 ),
                    2 * bVector[ 0 ] * bVector[ 1 ], 2 * bVector[ 0 ] * bVector[ 2 ],
                    2 * bVector[ 1 ] * bVector[ 2 ]
                ];

                //Add the row B row
                B.push( bRow );
            }

            //Return the B matrix
            return B;
        },

        /*Get the signals 1 to N at x, y, z
          devided by the corresponding bvalues */
        getS: function ( x, y, z, volumesData ) {
            var S = [],
                volumesDataLength = volumesData.length;

            for( var i = 1;  i < volumesDataLength; i++ ) {
                S.push( volumesData[ i ].get( x, y, z ) );
            }

            //Return the signals vector
            return S;
        },

        //Get the signal 0 at x, y, z
        getS0: function ( x, y, z, volumesData ) {
            return volumesData[ 0 ].get( x, y, z );
        },

        //Unique Coefficient to Tensor
        coefToTensor: function ( coef ) {
            return [
                [      coef[ 0 ],  coef[ 3 ] / 2, coef[ 4 ] / 2 ],
                [  coef[ 3 ] / 2,      coef[ 1 ], coef[ 5 ] / 2 ],
                [  coef[ 4 ] / 2,  coef[ 5 ] / 2, coef[ 2 ] ]
            ];
        },

        //Get the Max Eigen Vector from the eigen values
        getMaxEigenVector: function ( eigenVectors, eigenValues ) {
            var maxEigenValue = Math.max.apply( Math, eigenValues ),
                maxIndex = eigenValues.indexOf( maxEigenValue );
            return eigenVectors[ maxIndex ];
        },

        getDForVoxel: function ( x, y, z, volumesData, volumeMask, B, bValues, threshold ) {
                //The S0 signal
            var S0 = this.getS0( x, y, z, volumesData ),
                //The S signal
                S = this.getS( x, y, z, volumesData );
            /*Verify if the voxel is not mask filtered, 
              and S0, S signals are not 0 at the vocel*/
            if( ( !volumeMask || volumeMask.get( x, y, z ) ) && 
                    S0 >= threshold && numeric.all( S ) ) {

                    //Y = -Ln( S/ S0 ) 
                var Y = numeric.log( numeric.div( S, S0 ) ),
                    //Transpose B
                    transB = numeric.transpose( B ),
                    //( Bt * B ) ^ (-1)
                    invBbyTransB = numeric.inv( numeric.dot( transB, B ) ),
                    //Retrieve X = Unique Tensor Coefficients
                    X = numeric.dot( -1 / bValues[ 1 ], numeric.dot( numeric.dot( invBbyTransB, transB ), Y ) ),
                    //Transform Unique Coefficients to a Tensor form
                    D = this.coefToTensor( X );
                
                //Return The Diffusion Tensor 
                return D;
            }
        },

        //Launch the Tensor estimation
        estimateTensor: function () {
            var volumesData = this.volumesData,
                Volume3 = this.NS( 'dMRIJS.Volume3' ),
                Volume33 = this.NS( 'dMRIJS.Volume33' ),
                volumeMask = this.volumeMask,
                volume = volumesData[ 0 ],
                dimX = volume.dimX(),
                dimY = volume.dimY(),
                dimZ = volume.dimZ(),
                bValues = this.bValues,
                threshold = this.threshold,
                DVoxel,
                D = Volume33.create( dimX, dimY, dimZ, Float64Array ),
                eigenVectors = Volume33.create( dimX, dimY, dimZ, Float64Array ),
                maxVectors = Volume3.create( dimX, dimY, dimZ, Float64Array ),
                eigenValues = Volume3.create( dimX, dimY, dimZ, Float64Array ),
                eigen;

            //Construct the B matrix (g * gT)
            var B = this.B( this.bVectors );

            //Construct the D tensor, eigenValues and eigenVectors
            for( var x = 0; x < dimX; x++ ) {
                for( var y = 0; y < dimY; y++ ) {
                    for( var z = 0; z < dimZ; z++ ) {
                        //D estimation
                        DVoxel = this.getDForVoxel( x, y, z, volumesData, volumeMask, 
                            B, bValues, threshold );

                        //If D is defined
                        if( DVoxel ) {
                            //Compute eigen valus and vectors
                            try {
                                //D Eigen values and vectors
                                eigen = numeric.eig( DVoxel );

                                //D voxel
                                D.set( x, y, z, DVoxel );

                                //D Eigen values
                                eigenValues.set( x, y, z, eigen.lambda.x );
                                
                                //D Eigen vectors
                                eigenVectors.set( x, y, z, eigen.E.x );
                                
                                //Max vectors
                                maxVectors.set( x, y, z, 
                                    this.getMaxEigenVector( eigen.E.x, eigen.lambda.x ) );
                            } catch(e) {
                            }
                        }
                    }
                }

                //Notify progress
                this.notify({
                    progress: Math.round( ( x + 1 ) / dimX  * 100 ),
                    desc: 'Tensor Estimation Slice :' + (x + 1) + ' / ' + dimX
                });
            }

            //Create a result object
            var result = {
                D: D, 
                eigenValues: eigenValues,
                eigenVectors: eigenVectors,
                maxVectors: maxVectors
            };

            //Return the result
            return result;
        },

        //Run the Thread
        run: function () {
            //Launch the estimation
            var result = this.estimateTensor();
            
            //Set the computation result
            this.setResult( result );

            //Terminate the Thread
            this.terminate();
        }
    }).mixin( myThreadJS.Thread )
    .dependsOn({
        script: [ numericURL ]
    });

    //The Parallel Version of the Tensor Estimation for LS method class
    var TensorEstimationParaLS = myThreadJS.BaseObject.extend({

        //Class constructor
        constructor: function ( data ) {

            var bValues = data.bval,
                bVecs = data.bvec,
                threshold = data.threshold,
                volumes = data.volumes,
                mask = data.mask,
                splitNumber = data.split_number,
                self = this;

            //Create the Tensor Estimation computing deferred and promise
            var deferred = when.defer();
            this.deferred = deferred;
            this.promise = this.deferred.promise;

            //Launch the Tensor Estimation
            this.estimateTensor( volumes, mask, threshold, 
                bValues, bVecs, splitNumber );
        },

        //Estimation the diffusion tensor
        estimateTensor: function ( volumes, mask, threshold, 
                bValues, bVecs, splitNumber ) {
            var self = this,
                volumesNumber = bValues.length;

            //Split the volumes and the mask
            return this.splitVolumesAndMask( splitNumber, volumesNumber, 
                    volumes, mask ).then(function ( splitsResult ) {
                //Parallel estimation of the diffusion tensor
                return self.parallelEstimation( splitsResult[ 0 ], splitsResult[ 1 ],
                    splitNumber, bValues, bVecs, threshold, volumes[ 0 ].dimX() );
            }).then(function ( tensorEstimations ) {
                //Join the tensor estimations subresults
                return self.joinResults( tensorEstimations, splitNumber );
            }).then(function ( joinResults ) {
                //Set the Tensor Estimation result
                self.deferred.resolve({
                    D: joinResults[ 0 ],
                    eigenValues: joinResults[ 1 ],
                    eigenVectors: joinResults[ 2 ],
                    maxVectors: joinResults[ 3 ]
                });
            }).catch(function ( error ) {
                self.deferred.reject( error );
            });
        },

        //Split volumes and mask into subvolumes
        splitVolumesAndMask: function ( splitNumber, volumesNumber, volumes, mask ) {
            var self = this,
                volumeSplitPromises = [],
                volumesSplitNumber = 0;

            //Split the volumes
            for( var i = 0; i < volumesNumber; i++ ) {
                var volumeSplitPromise = volumes[ i ].split( splitNumber );
                volumeSplitPromise.then(function () {
                    volumesSplitNumber++;
                    self.deferred.notify({
                        desc: 'Split Volumes and Mask Before Tensor Estimation : ' + volumesSplitNumber + '/' + (volumesNumber + 1),
                        progress: Math.round( ( volumesSplitNumber / ( ( volumesNumber + 1 ) * 3 ) ) * 100 )
                    });
                });
                volumeSplitPromises.push( volumeSplitPromise );
            }

            //Split the mask
            var splitMaskPromise = mask && mask.split( splitNumber );
            splitMaskPromise && splitMaskPromise.then(function () {
                volumesSplitNumber++;
                self.deferred.notify({
                    desc: 'Split Volumes and Mask Before Tensor Estimation : ' + volumesSplitNumber + '/' + (volumesNumber + 1),
                    progress: Math.round( ( volumesSplitNumber / ( ( volumesNumber + 1 ) * 3 ) ) * 100 )
                });
            });
      
            //Wait for split results
            return when.join( when.all( volumeSplitPromises ), 
                splitMaskPromise );
      
        },

        //Launch the parallel tensor estimation
        parallelEstimation: function ( volumesSplits, maskSplits, splitNumber,
                bValues, bVecs, threshold, dimX ) {
            var self = this,
                threadPromises = [],
                estimationProgress = [],
                volumesNumber = bValues.length;

            for( var i = 0; i < splitNumber; i++ ) {
                estimationProgress[ i ] = 0;
            }

            for( var i = 0; i < splitNumber; i++ ) {
                var volumesSplit = [],
                    maskSplit = maskSplits && maskSplits[ i ];
                    
                //Volumes Split data
                for( var j = 0; j < volumesNumber; j++ ) {
                    volumesSplit.push( volumesSplits[ j ][ i ] );
                }

                //Launch the tensor estimation
                var tensorEstimationSeqLS = new TensorEstimationSeqLS({
                    bval: bValues,
                    bvec: bVecs,
                    volumes: volumesSplit,
                    mask: maskSplit,
                    threshold: threshold
                });

                (function ( tensorEstimationPromise, index ) {
                    tensorEstimationPromise.then( undefined, undefined, function ( notif ) {
                        if( !notif.progress ) {
                            return;
                        }
                        var sum = 0,
                            estimationProgressLength = estimationProgress.length,
                            newNotif = {};
                        estimationProgress[ index ] = notif.progress / splitNumber;
                        for( var i = 0; i < estimationProgressLength; i++ ) {
                            sum += estimationProgress[ i ];
                        }

                        newNotif.progress = Math.round( sum / 3 + 33 );
                        newNotif.desc = 'Tensor Estimation Slice :' + Math.round( sum * dimX / 100 ) + ' / ' + dimX;
                        self.deferred.notify( newNotif );
                    });
                })( tensorEstimationSeqLS.getPromise(), i );

                //Push the tensor estimation promise
                threadPromises.push( tensorEstimationSeqLS.getPromise() );
            }

            //Wait for the termination of threads
            return when.all( threadPromises );            
        },

        //Join D Tensor Splits
        joinDSplits: function ( DSplits, joinPromises ) {
            var DSplitsJoinPromise = dMRIJS.Volume33.join( DSplits ),
                self = this;
            DSplitsJoinPromise.then(function () {
                self.joinPromiseNumber++;
                self.deferred.notify({
                    desc: 'Join Tensor Results : ' + self.joinPromiseNumber + '/' + 4,
                    progress: (self.joinPromiseNumber / (4 * 3)) * 100 + 66 
                });
            });
            joinPromises[ 0 ] = DSplitsJoinPromise; 
            return DSplitsJoinPromise;
        },

        //Join eigen values Splits
        joinEValSplits: function ( eValSplits, joinPromises ) {
            var eValSplitsJoinPromise = dMRIJS.Volume3.join( eValSplits ),
                self = this;
            eValSplitsJoinPromise.then(function () {
                self.joinPromiseNumber++;
                self.deferred.notify({
                    desc: 'Join Tensor Results : ' + self.joinPromiseNumber + '/' + 4,
                    progress: (self.joinPromiseNumber / (4 * 3)) * 100 + 66
                });
            });
            joinPromises[ 1 ] = eValSplitsJoinPromise;

            return eValSplitsJoinPromise;
        },

        //Join eigen vectors splits
        joinEVecSplits: function ( eVecSplits, joinPromises ) {
            var eVecSplitsJoinPromise = dMRIJS.Volume33.join( eVecSplits ),
                self = this;
            eVecSplitsJoinPromise.then(function () {
                self.joinPromiseNumber++;
                self.deferred.notify({
                    desc: 'Join Tensor Results : ' + self.joinPromiseNumber + '/' + 4,
                    progress: (self.joinPromiseNumber / (4 * 3)) * 100 + 66
                });
            });
            joinPromises[ 2 ] = eVecSplitsJoinPromise;
            return eVecSplitsJoinPromise;
        },

        //join max vectors splits
        joinMaxVecsSplits: function ( maxVecSplits, joinPromises ) {
            var maxVecSplitsJoinPromise = dMRIJS.Volume3.join( maxVecSplits ),
                self = this;
            maxVecSplitsJoinPromise.then(function () {
                self.joinPromiseNumber++;
                self.deferred.notify({
                    desc: 'Join Tensor Results : ' + self.joinPromiseNumber + '/' + 4,
                    progress: (self.joinPromiseNumber / (4 * 3)) * 100 + 66 
                });
            });
            joinPromises[ 3 ] = maxVecSplitsJoinPromise;
            return maxVecSplitsJoinPromise;
        },

        //Join tensor estimation results
        joinResults: function ( tensorEstimations ) {
            var self = this,
                splitNumber = tensorEstimations.length,
                DSplits = [],
                eValSplits = [],
                eVecSplits = [],
                maxVecSplits = [],
                joinPromises = [],
                joinPromiseNumber = 0;

            this.joinPromiseNumber = 0;

            //Extract splits
            for( var i = 0; i < splitNumber; i++ ) {
                var tensorEstimation = tensorEstimations[ i ];
                DSplits.push( tensorEstimation.D );
                eValSplits.push( tensorEstimation.eigenValues );
                eVecSplits.push( tensorEstimation.eigenVectors );
                maxVecSplits.push( tensorEstimation.maxVectors );
            }

            try {
            
            /*Launch results joining*/
            if( $$.isMobile() ) {
                /* On Mobile, join volumes sequentially */
                //DSplits Join
                return this.joinDSplits( DSplits, joinPromises ).then(function () {
                    //eValSplits Join
                    self.joinEValSplits( eValSplits, joinPromises );
                }).then(function () {
                    //eVecSplits Join
                    self.joinEVecSplits( eVecSplits, joinPromises );
                }).then(function () {
                    //maxVecSplits Join
                    self.joinMaxVecsSplits( maxVecSplits, joinPromises );
                }).then(function () {
                    //Join the results through a promise
                    return when.all( joinPromises );                
                });
            } else {
                /* On Desktop, join volumes in prallel */
                //DSplits Join
                this.joinDSplits( DSplits, joinPromises );

                //eValSplits Join
                this.joinEValSplits( eValSplits, joinPromises );
                
                //eVecSplits Join
                this.joinEVecSplits( eVecSplits, joinPromises );
                
                //maxVecSplits Join
                this.joinMaxVecsSplits( maxVecSplits, joinPromises );

                //Waiting for all joining results
                return when.all( joinPromises );
            }
            

            } catch(e) {

               alert(e.toString());
            }

        },

        //Get computation promise
        getPromise: function () {
            return this.promise;
        }
    });

    //Return the DTIEstimation LS Algorithms.
    return {
        TensorEstimationSeqLS: TensorEstimationSeqLS,
        TensorEstimationParaLS: TensorEstimationParaLS
    };
});