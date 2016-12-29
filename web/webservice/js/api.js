if (typeof qbb == "undefined"){
	var qbb = {};
}

(function($) {
	if(typeof qbb.inf == "undefined") {

		qbb.inf = {
			service_url: "http://napeasy.org/napeasy_api/api",

			isValidJobId: function(s){
				return s.match(/[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}/ig);
			},

            createHTJob: function(user, pmcids, searchCB, errorCB){
                var apiName = "createHTJob";
                var sendObject={
                    r:apiName,
					user: user,
                    pmcids: pmcids
                };
                qbb.inf.callAPI(sendObject, searchCB, errorCB);
            },

            getJobDetail: function(jobId, searchCB, errorCB){
                var apiName = "getJobDetail";
                var sendObject={
                    r:apiName,
                    jobId: jobId
                };
                qbb.inf.callAPI(sendObject, searchCB, errorCB);
            },

            getPaperFullText: function(pmcid, searchCB){
                var apiName = "getPaperFullText";
                var sendObject={
                    r:apiName,
                    pmcid: pmcid
                };
                qbb.inf.callAPI(sendObject, searchCB);
            },

			callAPI: function(sendObject, cb, error){
				qbb.inf.ajax.doPost(sendObject, function(s){
					var ret = s;
					if (ret && ret.status == "200" && ret.data)
					{
						if (typeof cb == 'function')
							cb(ret.data);
					}else
					{
						if (typeof cb == 'function')
							cb();
					}
				}, function(jqXHR, textStatus, errorThrown){
					if (typeof error == 'function')error(jqXHR, textStatus, errorThrown);
				});
			},

			ajax: {
					doGet:function(sendData,success,error){
						qbb.inf.ajax.doSend("Get",null,sendData,success,error);
					},
					doPost:function(sendData,success,error){
						qbb.inf.ajax.doSend("Post",null,sendData,success,error);
					},
					doSend:function(method,url,sendData,success,error){
						dataSuccess = function(data){
							(success)(eval(data));
						};
						if (sendData) sendData.token = "";
						jQuery.ajax({
							   type: method || "Get",
							   url: url || qbb.inf.service_url,
							   data: sendData || [],
							   cache: false,
							   dataType: "jsonp", /* use "html" for HTML, use "json" for non-HTML */
							   success: dataSuccess /* (data, textStatus, jqXHR) */ || null,
							   error: error /* (jqXHR, textStatus, errorThrown) */ || null
						});
					}
			}
		};
	}
})(jQuery);