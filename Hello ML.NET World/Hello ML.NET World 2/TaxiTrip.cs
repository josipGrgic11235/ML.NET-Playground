using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Hello_ML.NET_World_2
{
    public class TaxiTrip
    {
        /// <summary>
        /// The ID of the taxi vendor
        /// </summary>
        [LoadColumn(0)]
        public string VendorId;

        /// <summary>
        /// The rate type of the taxi trip
        /// </summary>
        [LoadColumn(1)]
        public string RateCode;

        /// <summary>
        /// The number of passengers on the trip
        /// </summary>
        [LoadColumn(2)]
        public float PassengerCount;

        /// <summary>
        /// The amount of time the trip took
        /// </summary>
        [LoadColumn(3)]
        public float TripTime;

        /// <summary>
        /// The distance of the trip
        /// </summary>
        [LoadColumn(4)]
        public float TripDistance;

        /// <summary>
        /// The payment method (cash or credit card)
        /// </summary>
        [LoadColumn(5)]
        public string PaymentType;

        /// <summary>
        /// The total taxi fare paid
        /// </summary>
        [LoadColumn(6)]
        public float FareAmount;
    }
}
