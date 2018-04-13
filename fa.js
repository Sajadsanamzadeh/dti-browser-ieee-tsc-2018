//The Fractional Anisotropy algorithms module
define([
    'dMRIJS/volume',
    'mythread',
    'numeric'
], function ( dMRIJS, myThreadJS ) {

        //Get the Namespace module
    var BaseObject = myThreadJS.BaseObject,
        Thread = myThreadJS.Thread,
        //Get the numeric module URL
        numericURL = require.toUrl( 'numeric' ) + '.js';

    //The Sequential Version of the Fractional Anisotropy class
    var FASeq = BaseObject.extend( 'dMRIJS.FASeq', {
        
        //Class constructor
        constructor: function ( data ) {

            /*Initialize the parameters of the FA computation*/
            this.eigenValues = data.eigenValues;

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

        avgEigen: function ( eigenValues ) {
            var length  = eigenValues.length,
                sum = 0;

            //Sum the eigen values
            for(var i = 0; i < length; i++ ) {
                sum +=  eigenValues[ i ];
            }

            //Return the averate
            return sum/length;
        },

        varEigen: function( eigenValues ) {
            var length = eigenValues.length,
                avg = this.avgEigen( eigenValues ),
                squaredDiffEigenArray = [];

            //Sum the eigen values
            for(var i = 0; i < length; i++ ) {
                squaredDiffEigenArray.push( Math.pow( eigenValues[ i ] - avg, 2 ) );
            }

            //Compute the sum of squared diff and return it
            return squaredDiffEigenArray;
        },

        //Get FA From Eigen Values
        getFAFromEigenValues: function( eigenValues ) {
            var norm = numeric.norm2( eigenValues );

            if( norm > 0 ) {
                var varEigen = this.varEigen( eigenValues ),
                    sumVarEigen = numeric.sum( varEigen );
                return Math.sqrt( ( 3 / 2 ) * sumVarEigen ) / norm;
            } else {
                //FA = 0
                return 0;
            }
        },

        //Compute the Fractional Anisotropy
        computeFA: function () {
            var VolumeSpace = this.NS( 'myThreadJS.VolumeSpace' ),
                Volume3 = this.NS( 'dMRIJS.Volume3' ),
                faValue,
                min = Infinity,
                max = -Infinity,
                eigenValues = this.eigenValues,
                dimX = eigenValues.dimX(),
                dimY = eigenValues.dimY(),
                dimZ = eigenValues.dimZ(),
                FA = VolumeSpace.create( dimX, dimY, dimZ, Float64Array );

            for(var x = 0; x < dimX; x++) {
                for( var y = 0; y < dimY; y++ ) {
                    for( var z = 0; z < dimZ; z++ ) {
                        //Compute FA value for the voxel
                        faValue = this.getFAFromEigenValues( 
                            eigenValues.get( x, y, z ) );

                        //FA value shouldn't surpass 1
                        faValue = Math.min( 1, faValue );
                        
                        //Set the FA voxel value
                        FA.set( x, y, z, faValue );

                        //Compute the minimum and maximum values
                        min = Math.min( faValue, min );
                        max = Math.max( faValue, max );
                    }
                }

                //Notify progress
                this.notify({
                    progress: Math.round( ( x + 1 ) / dimX  * 100 ),
                    desc: 'Fractional Anisotropy Computation Slice :' + (x + 1) + ' / ' + dimX 
                });
            }

            //Create a result object
            var result = {
                FA: FA,
                min: min,
                max: max,
                eigenValues: eigenValues
            }

            //Return the result
            return result;
        },

        //Run the Thread
        run: function () {
            //Launch the FA computation
            var result = this.computeFA();

            //Set the computation result
            this.setResult( result );

            //Terminate the Thread
            this.terminate();
        }
    }).mixin( Thread )
    .dependsOn({
        script: [ numericURL ]
    });

    var FAColourSeq =  FASeq.extend( 'dMRIJS.FAColourSeq', {   
        //Class constructor
        constructor: function ( data ) {
            //Call the FASeq base constructor
            this.$super( 'dMRIJS.FASeq', this )( data );

            //Set the max vectors
            this.maxVectors = data.maxVectors;
        },

        computeFAColour: function ( FA ) {
            var maxVectors = this.maxVectors,
                Volume3 = this.NS( 'dMRIJS.Volume3' ),
                dimX = FA.dimX(),
                dimY = FA.dimY(),
                dimZ = FA.dimZ(),
                FAColour = Volume3.create( dimX, dimY, dimZ, Float64Array );

            for( var x = 0; x < dimX; x++ ) {
                for( var y = 0; y < dimY; y++ ) {
                    for( var z = 0; z < dimZ; z++ ) {
                        var maxVectorVal = maxVectors.get( x, y, z),
                            FAVal = FA.get( x, y, z );
                        FAColour.set( x, y, z, [ FAVal * Math.abs( maxVectorVal[ 0 ] ),
                            FAVal * Math.abs( maxVectorVal[ 1 ] ), FAVal * Math.abs( maxVectorVal[ 2 ] ) ] );
                    }
                }
            }

            return FAColour;
        }, 

        run: function () {
            //Launch the FA computation
            var resultFA = this.computeFA();

            //Launch the FA Colour Computation
            resultFA.FA = this.computeFAColour( resultFA.FA );

            //Set the computation result
            this.setResult( resultFA );

            //Terminate the Thread
            this.terminate();
        }
    });
 
    return {FASeq: FASeq, FAColourSeq: FAColourSeq};
});
